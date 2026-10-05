/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Chloride Project

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/matrix.hpp>
#include <ql/methods/finitedifferences/finitedifferencemodel.hpp>
#include <ql/methods/finitedifferences/meshers/concentrating1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholesstripsolver.hpp>
#include <ql/methods/finitedifferences/stepcondition.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace QuantLib {

    namespace {

        // FiniteDifferenceModel over a (nodes x options) matrix.  The
        // model is built from an evolver, so operator_type and bc_set are
        // never used.
        struct StripTraits {
            typedef FdmLinearOp operator_type;
            typedef Matrix array_type;
            typedef std::vector<int> bc_set;
            typedef StepCondition<Matrix> condition_type;
        };

        // DouglasScheme::step for a 1-D operator, every column at once:
        // y = a + dt L a, rhs = y - theta dt L a, solve (1 - theta dt L).
        class StripDouglasScheme {
          public:
            typedef StripTraits traits;
            typedef traits::array_type array_type;

            StripDouglasScheme(Real theta, ext::shared_ptr<FdmBlackScholesOp> op)
            : dt_(Null<Real>()), theta_(theta), op_(std::move(op)) {}

            void setStep(Time dt) { dt_ = dt; }

            void step(Matrix& a, Time t) {
                QL_REQUIRE(t-dt_ > -1e-8, "a step towards negative time given");
                op_->setTime(std::max(0.0, t-dt_), t);
                const TripleBandLinearOp& map = op_->map();

                const Matrix la = map.apply_columns(a);
                const Real thetaDt = theta_*dt_;
                Matrix rhs(a.rows(), a.columns());
                for (Size i=0; i < a.rows(); ++i) {
                    const Real* ai = a.row_begin(i);
                    const Real* li = la.row_begin(i);
                    Real* ri = rhs.row_begin(i);
                    for (Size j=0; j < a.columns(); ++j) {
                        const Real y = ai[j] + dt_*li[j];
                        ri[j] = y - thetaDt*li[j];
                    }
                }
                a = map.solve_splitting_columns(rhs, -theta_*dt_);
            }

          private:
            Time dt_;
            const Real theta_;
            const ext::shared_ptr<FdmBlackScholesOp> op_;
        };

        // The vanilla engine's composite, in its order: FdmDividendHandler
        // (1-D, spot model), FdmAmericanStepCondition, then
        // Fdm1DimSolver's FdmSnapshotCondition.
        class StripStepCondition : public StepCondition<Matrix> {
          public:
            StripStepCondition(
                std::vector<Real> spots,
                std::vector<std::pair<Time, ext::shared_ptr<Dividend> > > dividends,
                Matrix exercise,
                bool american,
                Time snapshotTime)
            : s_(std::move(spots)), dividends_(std::move(dividends)),
              exercise_(std::move(exercise)), american_(american),
              snapshotTime_(snapshotTime) {}

            void applyTo(Matrix& a, Time t) const override {
                // FdmDividendHandler applies the first dividend whose time
                // matches exactly
                for (const auto& d : dividends_) {
                    if (d.first == t) {
                        applyDividend(a, *d.second);
                        break;
                    }
                }
                if (american_) {
                    for (Size i=0; i < a.rows(); ++i) {
                        const Real* e = exercise_.row_begin(i);
                        Real* v = a.row_begin(i);
                        for (Size j=0; j < a.columns(); ++j)
                            if (e[j] > v[j])
                                v[j] = e[j];
                    }
                }
                if (t == snapshotTime_)
                    snapshot_ = a;
            }

            const Matrix& snapshot() const { return snapshot_; }

          private:
            // LinearInterpolation(s, column)(max(s[0], s[k] - D), true)
            // for every column; the bracket depends on the node only.
            void applyDividend(Matrix& a, const Dividend& div) const {
                const Matrix copy(a);
                const Size n = s_.size();
                for (Size k=0; k < n; ++k) {
                    const Real x = std::max(s_[0], s_[k] - div.amount(s_[k]));
                    Size i;
                    if (x < s_[0])
                        i = 0;
                    else if (x > s_[n-1])
                        i = n-2;
                    else
                        i = std::upper_bound(s_.begin(), s_.end()-1, x) - s_.begin() - 1;
                    const Real dx = x - s_[i];
                    const Real h = s_[i+1] - s_[i];
                    const Real* y0 = copy.row_begin(i);
                    const Real* y1 = copy.row_begin(i+1);
                    Real* out = a.row_begin(k);
                    for (Size j=0; j < a.columns(); ++j)
                        out[j] = y0[j] + dx*((y1[j]-y0[j])/h);
                }
            }

            const std::vector<Real> s_;
            const std::vector<std::pair<Time, ext::shared_ptr<Dividend> > > dividends_;
            const Matrix exercise_;
            const bool american_;
            const Time snapshotTime_;
            mutable Matrix snapshot_;
        };


        // Exact average of a plain-vanilla payoff over node i's ln S cell
        // [x_i - dminus/2, x_i + dplus/2] (FdmCellAveragingInnerValue's
        // cell); boundary nodes take the payoff at the node, as there.
        Real exactCellAverage(const StrikedTypePayoff& payoff,
                              const std::vector<Real>& x, Size i) {
            const Size n = x.size();
            if (i == 0 || i == n-1)
                return payoff(std::exp(x[i]));
            const Real a = x[i] - (x[i] - x[i-1])/2.0;
            const Real b = x[i] + (x[i+1] - x[i])/2.0;
            const Real K = payoff.strike();
            const Real L = std::log(K);
            const Real ea = std::exp(a), eb = std::exp(b);
            Real integral;
            if (payoff.optionType() == Option::Put) {
                if (b <= L)
                    integral = K*(b - a) - (eb - ea);
                else if (a >= L)
                    integral = 0.0;
                else
                    integral = K*(L - a) - (K - ea);
            } else {
                if (a >= L)
                    integral = (eb - ea) - K*(b - a);
                else if (b <= L)
                    integral = 0.0;
                else
                    integral = (eb - K) - K*(b - L);
            }
            return integral/(b - a);
        }
    }

    FdmBlackScholesStripSolver::FdmBlackScholesStripSolver(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process,
        const Date& maturityDate,
        std::vector<ext::shared_ptr<StrikedTypePayoff> > payoffs,
        bool american,
        DividendSchedule dividends,
        ext::shared_ptr<Fdm1dMesher> mesher,
        Size tGrid,
        Real illegalLocalVolOverwrite,
        std::vector<Time> stoppingTimes,
        bool exactCellAverage)
    : process_(std::move(process)), maturityDate_(maturityDate),
      payoffs_(std::move(payoffs)), american_(american),
      dividends_(std::move(dividends)), mesher_(std::move(mesher)),
      tGrid_(tGrid), illegalLocalVolOverwrite_(illegalLocalVolOverwrite),
      stoppingTimes_(std::move(stoppingTimes)),
      exactCellAverage_(exactCellAverage) {
        QL_REQUIRE(process_, "null process");
        QL_REQUIRE(mesher_, "null mesher");
        QL_REQUIRE(!payoffs_.empty(), "no payoffs given");
        for (const auto& p : payoffs_) {
            QL_REQUIRE(p, "null payoff");
            QL_REQUIRE(!exactCellAverage_
                       || ext::dynamic_pointer_cast<PlainVanillaPayoff>(p),
                       "the exact cell average needs plain-vanilla payoffs");
        }
        QL_REQUIRE(tGrid_ > 0, "at least one time step required");
    }

    FdmBlackScholesStripResults FdmBlackScholesStripSolver::solve() const {
        const Date refDate = process_->riskFreeRate()->referenceDate();
        const DayCounter dc = process_->riskFreeRate()->dayCounter();
        const Time maturity = process_->time(maturityDate_);
        QL_REQUIRE(maturity > 0.0, "maturity " << maturityDate_
                   << " is not after the reference date " << refDate);

        const auto mesher = ext::make_shared<FdmMesherComposite>(mesher_);
        const Size n = mesher->layout()->size();
        const Size m = payoffs_.size();
        const std::vector<Real>& x = mesher_->locations();

        // terminal values (cell averages) and exercise values, from the
        // vanilla engine's own inner-value calculator
        Matrix v(n, m), exercise(n, m);
        for (Size j=0; j < m; ++j) {
            FdmLogInnerValue calculator(payoffs_[j], mesher, 0);
            for (const auto& iter : *mesher->layout()) {
                exercise[iter.index()][j] = calculator.innerValue(iter, maturity);
                if (!exactCellAverage_)
                    v[iter.index()][j] = calculator.avgInnerValue(iter, maturity);
            }
            if (exactCellAverage_)
                for (Size i=0; i < n; ++i)
                    v[i][j] = exactCellAverage(*payoffs_[j], x, i);
        }

        // stopping times as FdmStepConditionComposite::vanillaComposite,
        // the engine's extra stops and Fdm1DimSolver's theta snapshot
        std::vector<std::pair<Time, ext::shared_ptr<Dividend> > > dividends;
        std::set<Time> stops;
        for (const auto& d : dividends_) {
            if (d->date() >= refDate && d->date() <= maturityDate_) {
                const Time t = dc.yearFraction(refDate, d->date());
                dividends.emplace_back(t, d);
                stops.insert(std::min(maturity, t));
                stops.insert(std::min(maturity, t + 1e-5));
            }
        }
        for (Time t : stoppingTimes_)
            if (t > 0.0 && t < maturity)
                stops.insert(t);
        const Time snapshotTime =
            0.99 * std::min(1.0/365.0, stops.empty() ? maturity : *stops.begin());
        stops.insert(snapshotTime);

        std::vector<Real> spots(n);
        for (Size i=0; i < n; ++i)
            spots[i] = std::exp(x[i]);
        const StripStepCondition condition(
            spots, dividends, exercise, american_, snapshotTime);

        const auto op = ext::make_shared<FdmBlackScholesOp>(
            mesher, process_, payoffs_.front()->strike(),
            true, illegalLocalVolOverwrite_, 0);
        FiniteDifferenceModel<StripDouglasScheme> model(
            StripDouglasScheme(0.5, op),
            std::vector<Time>(stops.begin(), stops.end()));
        model.rollback(v, maturity, 0.0, tGrid_, condition);

        // Fdm1DimSolver's read-out: monotonic natural cubic spline in ln S
        const Real spot = process_->x0();
        const Real x0 = std::log(spot);
        FdmBlackScholesStripResults results;
        results.value.resize(m);
        results.delta.resize(m);
        results.gamma.resize(m);
        results.theta.resize(m);
        const Matrix& snapshot = condition.snapshot();
        Array column(n), thetaColumn(n);
        for (Size j=0; j < m; ++j) {
            for (Size i=0; i < n; ++i) {
                column[i] = v[i][j];
                thetaColumn[i] = snapshot[i][j];
            }
            const MonotonicCubicNaturalSpline spline(x.begin(), x.end(), column.begin());
            const Real d1 = spline.derivative(x0);
            const Real d2 = spline.secondDerivative(x0);
            results.value[j] = spline(x0);
            results.delta[j] = d1/spot;
            results.gamma[j] = (d2 - d1)/(spot*spot);
            const MonotonicCubicNaturalSpline thetaSpline(
                x.begin(), x.end(), thetaColumn.begin());
            // Fdm1DimSolver::thetaAt has no snapshot to difference
            // against when the first stopping time is today
            results.theta[j] = snapshotTime == 0.0 ?
                Null<Real>() : (thetaSpline(x0) - results.value[j]) / snapshotTime;
        }
        results.illegalLocalVolCount = op->illegalLocalVolCount();
        return results;
    }

    ext::shared_ptr<Fdm1dMesher> FdmBlackScholesStripSolver::stripMesher(
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
        Time maturity,
        const std::vector<Real>& strikes,
        const DividendSchedule& dividends,
        Size xGrid,
        Real scaleFactor,
        Real eps,
        Real spotDensity) {
        QL_REQUIRE(!strikes.empty(), "no strikes given");
        Real xMin = QL_MAX_REAL, xMax = -QL_MAX_REAL;
        for (Real strike : strikes) {
            const std::pair<Real, Real> range = FdmBlackScholesMesher::xRange(
                process, maturity, strike, Null<Real>(), Null<Real>(),
                eps, scaleFactor, dividends);
            xMin = std::min(xMin, range.first);
            xMax = std::max(xMax, range.second);
        }
        const Real x0 = std::log(process->x0());
        QL_REQUIRE(xMin < x0 && x0 < xMax,
                   "spot outside the strip's mesh range");
        return ext::make_shared<Concentrating1dMesher>(
            xMin, xMax, xGrid, std::make_pair(x0, spotDensity), true);
    }

}
