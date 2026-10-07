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
#include <ql/methods/finitedifferences/meshers/gradedcore1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholesstripsolver.hpp>
#include <ql/methods/finitedifferences/stepcondition.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdividendhandler.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__)
#define QL_COLUMN_KERNEL __attribute__((target_clones("avx2", "default")))
#else
#define QL_COLUMN_KERNEL
#endif

namespace QuantLib {

    namespace {

        // rhs = (a + dt L a) - theta dt L a, in place over w = L a
        // (DouglasScheme's two Array expressions, element by element)
        QL_COLUMN_KERNEL
        void douglasRhs(const Matrix& a, Matrix& w, Real dt, Real thetaDt) {
            const Size m = a.columns();
            for (Size i=0; i < a.rows(); ++i) {
                const Real* __restrict ai = a.row_begin(i);
                Real* __restrict wi = w.row_begin(i);
                for (Size j=0; j < m; ++j) {
                    const Real la = wi[j];
                    const Real y = ai[j] + dt*la;
                    wi[j] = y - thetaDt*la;
                }
            }
        }

        // FdmAmericanStepCondition: a = max(a, exercise)
        QL_COLUMN_KERNEL
        void americanMax(Matrix& a, const Matrix& exercise) {
            const Size m = a.columns();
            for (Size i=0; i < a.rows(); ++i) {
                const Real* __restrict e = exercise.row_begin(i);
                Real* __restrict v = a.row_begin(i);
                for (Size j=0; j < m; ++j)
                    if (e[j] > v[j])
                        v[j] = e[j];
            }
        }

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

                // work = L a, then in place rhs = (a + dt L a) - theta dt L a
                map.apply_columns(a, work_);
                douglasRhs(a, work_, dt_, theta_*dt_);
                map.solve_splitting_columns(work_, -theta_*dt_);
                a.swap(work_);
            }

          private:
            Time dt_;
            const Real theta_;
            const ext::shared_ptr<FdmBlackScholesOp> op_;
            Matrix work_;
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
              snapshotTime_(snapshotTime) {
                jumps_.reserve(dividends_.size());
                for (const auto& d : dividends_)
                    jumps_.emplace_back(s_, *d.second);
            }

            void applyTo(Matrix& a, Time t) const override {
                // FdmDividendHandler applies the first dividend whose time
                // matches exactly
                for (Size d=0; d < dividends_.size(); ++d) {
                    if (dividends_[d].first == t) {
                        const Matrix copy(a);
                        jumps_[d].apply(copy, a);
                        break;
                    }
                }
                if (american_)
                    americanMax(a, exercise_);
                if (t == snapshotTime_)
                    snapshot_ = a;
            }

            const Matrix& snapshot() const { return snapshot_; }

          private:
            const std::vector<Real> s_;
            const std::vector<std::pair<Time, ext::shared_ptr<Dividend> > > dividends_;
            // FdmDividendHandler's jump, one per dividend (chloride #593)
            std::vector<detail::FdmDividendJump> jumps_;
            const Matrix exercise_;
            const bool american_;
            const Time snapshotTime_;
            mutable Matrix snapshot_;
        };


        // Nodes either side of spot that the delta/gamma read-out may use.
        const Size readoutHalfWidth = 2;

        // Fornberg's weights: w1[k], w2[k] give the first and second
        // derivative at 0 of the polynomial through (z[k], f[k]), k < m.
        void fornbergWeights(const Real* z, Size m, Real* w1, Real* w2) {
            Real c[2*readoutHalfWidth+1][3] = {};
            c[0][0] = 1.0;
            Real c1 = 1.0, c4 = z[0];
            for (Size i=1; i < m; ++i) {
                const Size mn = std::min<Size>(i, 2);
                Real c2 = 1.0;
                const Real c5 = c4;
                c4 = z[i];
                for (Size jj=0; jj < i; ++jj) {
                    const Real c3 = z[i] - z[jj];
                    c2 *= c3;
                    if (jj == i-1) {
                        for (Size k=mn; k >= 1; --k)
                            c[i][k] = c1*(Real(k)*c[i-1][k-1] - c5*c[i-1][k])/c2;
                        c[i][0] = -c1*c5*c[i-1][0]/c2;
                    }
                    for (Size k=mn; k >= 1; --k)
                        c[jj][k] = (c4*c[jj][k] - Real(k)*c[jj][k-1])/c3;
                    c[jj][0] = c4*c[jj][0]/c3;
                }
                c1 = c2;
            }
            for (Size k=0; k < m; ++k) {
                w1[k] = c[k][1];
                w2[k] = c[k][2];
            }
        }

        // dV/dS and d2V/dS2 at node i0 of column j: the polynomial in S
        // through the run of nodes within readoutHalfWidth of i0 that sit
        // on the same side of the exercise boundary as i0 (all exactly on
        // the exercise value, or all off it).  Away from the boundary that
        // is the quartic through five nodes; next to it the polynomial is
        // one-sided and never interpolates across the kink; inside the
        // exercise region it is exact (V = payoff, linear in S).  A
        // single boundary leaves at least three nodes in the run.
        std::pair<Real, Real> regionDerivatives(const std::vector<Real>& s,
                                                const Array& v,
                                                const Matrix& exercise,
                                                Size j, Size i0, bool american) {
            const auto onPayoff = [&](Size i) {
                return american && v[i] == exercise[i][j];
            };
            const bool side = onPayoff(i0);
            Size lo = i0, hi = i0;
            while (lo > i0 - readoutHalfWidth && onPayoff(lo-1) == side)
                --lo;
            while (hi < i0 + readoutHalfWidth && onPayoff(hi+1) == side)
                ++hi;
            const Size m = hi - lo + 1;
            QL_ENSURE(m >= 3, "fewer than three nodes on spot's side of the"
                      " exercise boundary");
            Real z[2*readoutHalfWidth+1], w1[2*readoutHalfWidth+1], w2[2*readoutHalfWidth+1];
            for (Size k=0; k < m; ++k)
                z[k] = s[lo+k] - s[i0];
            fornbergWeights(z, m, w1, w2);
            Real d1 = 0.0, d2 = 0.0;
            for (Size k=0; k < m; ++k) {
                d1 += w1[k]*v[lo+k];
                d2 += w2[k]*v[lo+k];
            }
            return std::make_pair(d1, d2);
        }


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
        // exercise value = payoff(exp(x_i)), FdmLogInnerValue::innerValue
        std::vector<Real> spots(n);
        for (Size i=0; i < n; ++i)
            spots[i] = std::exp(x[i]);
        Matrix v(n, m), exercise(n, m);
        for (Size j=0; j < m; ++j) {
            const Payoff& payoff = *payoffs_[j];
            for (Size i=0; i < n; ++i)
                exercise[i][j] = payoff(spots[i]);
            if (exactCellAverage_) {
                for (Size i=0; i < n; ++i)
                    v[i][j] = exactCellAverage(*payoffs_[j], x, i);
            } else {
                FdmLogInnerValue calculator(payoffs_[j], mesher, 0);
                for (const auto& iter : *mesher->layout())
                    v[iter.index()][j] = calculator.avgInnerValue(iter, maturity);
            }
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

        const StripStepCondition condition(
            spots, dividends, exercise, american_, snapshotTime);

        const auto op = ext::make_shared<FdmBlackScholesOp>(
            mesher, process_, payoffs_.front()->strike(),
            true, illegalLocalVolOverwrite_, 0);
        FiniteDifferenceModel<StripDouglasScheme> model(
            StripDouglasScheme(0.5, op),
            std::vector<Time>(stops.begin(), stops.end()));
        model.rollback(v, maturity, 0.0, tGrid_, condition);

        // Value and theta: Fdm1DimSolver's read-out, a monotonic natural
        // cubic spline in ln S.  Delta and gamma: the polynomial in S
        // through the nodes around spot that lie on spot's side of the
        // American exercise boundary (see regionDerivatives).  The spline's
        // derivatives are not usable near that boundary: the kink in V''
        // bends the spline across the neighbouring nodes, so at a spot
        // whose neighbours all sit exactly on the payoff it reports delta
        // < -1 and negative gamma, and it does not converge under
        // refinement.
        const Real spot = process_->x0();
        const Real x0 = std::log(spot);
        Size i0 = std::lower_bound(x.begin(), x.end(), x0) - x.begin();
        if (i0 == n || (i0 > 0 && x0 - x[i0-1] < x[i0] - x0))
            --i0;
        QL_REQUIRE(i0 >= readoutHalfWidth && i0 + readoutHalfWidth < n
                   && std::fabs(x[i0] - x0) < 1e-10,
                   "spot " << spot << " is not an interior node of the strip mesh;"
                   " use stripMesher(), which makes it one");
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
            results.value[j] = spline(x0);
            const std::pair<Real, Real> d = regionDerivatives(
                spots, column, exercise, j, i0, american_);
            results.delta[j] = d.first;
            results.gamma[j] = d.second;
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
        Real spotDensity,
        Real coreStdDevs,
        Real coreFraction) {
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
        if (coreStdDevs == Null<Real>())
            return ext::make_shared<Concentrating1dMesher>(
                xMin, xMax, xGrid, std::make_pair(x0, spotDensity), true);
        QL_REQUIRE(coreStdDevs > 0.0,
                   "core width must be positive, got " << coreStdDevs
                   << " standard deviations");
        return ext::make_shared<GradedCore1dMesher>(
            xMin, xMax, xGrid, x0,
            coreStdDevs * atmStdDev(process, maturity, dividends), coreFraction);
    }

    Real FdmBlackScholesStripSolver::atmStdDev(
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
        Time maturity,
        const DividendSchedule& dividends) {
        QL_REQUIRE(maturity > 0.0, "positive maturity required");
        const Handle<YieldTermStructure>& rTS = process->riskFreeRate();
        const Handle<YieldTermStructure>& qTS = process->dividendYield();
        // spot-model forward: S Q(T)/R(T) less each dividend carried
        // from its ex-date to T (FractionalDividend at the forward then)
        const Real growth = qTS->discount(maturity) / rTS->discount(maturity);
        Real forward = process->x0() * growth;
        for (const auto& d : dividends) {
            const Time t = process->time(d->date());
            if (t < 0.0 || t > maturity)
                continue;
            const Real carryToT = growth * rTS->discount(t) / qTS->discount(t);
            const Real forwardAtT = process->x0() * qTS->discount(t) / rTS->discount(t);
            forward -= d->amount(forwardAtT) * carryToT;
        }
        QL_REQUIRE(forward > 0.0, "non-positive forward " << forward);
        return process->blackVolatility()->blackVol(maturity, forward, true)
            * std::sqrt(maturity);
    }

}
