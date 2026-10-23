/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Brett Graves

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

#include <ql/math/matrixutilities/choleskydecomposition.hpp>
#include <ql/math/matrixutilities/svd.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/termstructures/yield/fomcstepcurve.hpp>
#include <ql/utilities/dataformatters.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace QuantLib {

    namespace {

        std::string isoDate(const Date& d) {
            std::ostringstream out;
            out << io::iso_date(d);
            return out.str();
        }

        Real safeRatio(Real num, Real den) { return den == 0.0 ? 0.0 : num / den; }

        //! all clamped B-spline basis values of degree p at x (Cox-de Boor)
        std::vector<Real> bsplineBasis(const std::vector<Real>& t, Size p, Real x) {
            Size m = t.size();
            std::vector<Real> n(m - 1, 0.0);
            for (Size i = 0; i + 1 < m; ++i)
                n[i] = (t[i] <= x && x < t[i + 1]) ? 1.0 : 0.0;
            for (Size d = 1; d <= p; ++d) {
                for (Size i = 0; i + d + 1 < m; ++i) {
                    Real left = safeRatio(x - t[i], t[i + d] - t[i]) * n[i];
                    Real right = safeRatio(t[i + d + 1] - x, t[i + d + 1] - t[i + 1]) * n[i + 1];
                    n[i] = left + right;
                }
            }
            n.resize(m - p - 1);
            return n;
        }

        //! first derivatives of the cubic basis at x, from the degree-2 basis
        std::vector<Real> cubicFirstDerivatives(const std::vector<Real>& t, Real x) {
            std::vector<Real> q = bsplineBasis(t, 2, x);  // size m - 3
            Size nSpline = t.size() - 4;
            std::vector<Real> d1(nSpline, 0.0);
            for (Size j = 0; j < nSpline; ++j)
                d1[j] = 3.0 * (safeRatio(q[j], t[j + 3] - t[j]) -
                               safeRatio(q[j + 1], t[j + 4] - t[j + 1]));
            return d1;
        }

        //! square-root factor R (R'R = G) of a Gram matrix, regularised by trace
        Matrix gramFactor(Matrix gram) {
            Real trace = 0.0;
            for (Size a = 0; a < gram.rows(); ++a)
                trace += gram[a][a];
            if (trace == 0.0)  // empty region (e.g. no tension past the curve's end)
                return Matrix(gram.rows(), gram.columns(), 0.0);
            for (Size a = 0; a < gram.rows(); ++a)
                gram[a][a] += 1e-12 * trace;
            return transpose(CholeskyDecomposition(gram));
        }

        //! second derivatives of the cubic basis at x, from the degree-1 basis
        std::vector<Real> cubicSecondDerivatives(const std::vector<Real>& t, Real x) {
            std::vector<Real> h = bsplineBasis(t, 1, x);  // size m - 2
            Size nSpline = t.size() - 4;
            std::vector<Real> d2(nSpline, 0.0);
            for (Size j = 0; j < nSpline; ++j) {
                Real a = safeRatio(h[j], t[j + 2] - t[j]) - safeRatio(h[j + 1], t[j + 3] - t[j + 1]);
                Real b = safeRatio(h[j + 1], t[j + 3] - t[j + 1]) -
                         safeRatio(h[j + 2], t[j + 4] - t[j + 2]);
                d2[j] = 6.0 * (safeRatio(a, t[j + 3] - t[j]) - safeRatio(b, t[j + 4] - t[j + 1]));
            }
            return d2;
        }

        Date firstOfNextMonth(const Date& d) {
            Month m = d.month();
            Year y = d.year();
            return m == December ? Date(1, January, y + 1) :
                                   Date(1, static_cast<Month>(static_cast<int>(m) + 1), y);
        }

        //! Residuals of the band problem on linearised quote errors
        class LinearBandCost : public CostFunction {
          public:
            LinearBandCost(const Array& e,
                           const Matrix& Je,
                           const Array& hw,
                           const Array& inner,
                           Real wall,
                           const Matrix& P,
                           const Array& p0,
                           const Array& x)
            : e_(e), Je_(Je), hw_(hw), inner_(inner), wall_(wall), P_(P), p0_(p0), x_(x) {}
            Real value(const Array& d) const override {
                Array v = values(d);
                return std::sqrt(DotProduct(v, v));
            }
            Array values(const Array& d) const override {
                Array ee = e_ + Je_ * d;
                Array pen = P_ * (x_ + d) + p0_;
                Size n = ee.size();
                Array out(2 * n + pen.size());
                for (Size i = 0; i < n; ++i) {
                    out[i] = ee[i] / inner_[i];
                    Real over = std::max(std::fabs(ee[i]) - hw_[i], 0.0);
                    out[n + i] = (ee[i] > 0.0 ? over : -over) / wall_;
                }
                std::copy(pen.begin(), pen.end(), out.begin() + 2 * n);
                return out;
            }

          private:
            Array e_;
            Matrix Je_;
            Array hw_, inner_;
            Real wall_;
            Matrix P_;
            Array p0_, x_;
        };

        Matrix stack(const Matrix& top, const Matrix& bottom) {
            Matrix out(top.rows() + bottom.rows(), top.columns());
            for (Size i = 0; i < top.rows(); ++i)
                std::copy(top.row_begin(i), top.row_end(i), out.row_begin(i));
            for (Size i = 0; i < bottom.rows(); ++i)
                std::copy(bottom.row_begin(i), bottom.row_end(i), out.row_begin(top.rows() + i));
            return out;
        }

        //! minimum-norm least squares through the SVD (singular values below
        //! eps * max(m, n) * s_max dropped): deterministic where the problem is
        //! nearly singular, e.g. tail forwards beyond the last instrument
        Array leastSquares(const Matrix& A, const Array& b) {
            SVD svd(A);
            const Array& s = svd.singularValues();
            const Matrix& U = svd.U();
            const Matrix& V = svd.V();
            Real sMax = *std::max_element(s.begin(), s.end());
            Real cutoff = QL_EPSILON * std::max(A.rows(), A.columns()) * sMax;
            Array x(A.columns(), 0.0);
            for (Size k = 0; k < s.size(); ++k) {
                if (s[k] <= cutoff)
                    continue;
                Real ub = 0.0;
                for (Size i = 0; i < A.rows(); ++i)
                    ub += U[i][k] * b[i];
                ub /= s[k];
                for (Size j = 0; j < A.columns(); ++j)
                    x[j] += V[j][k] * ub;
            }
            return x;
        }

        Array concat(const Array& a, const Array& b) {
            Array out(a.size() + b.size());
            std::copy(a.begin(), a.end(), out.begin());
            std::copy(b.begin(), b.end(), out.begin() + a.size());
            return out;
        }

    }

    // ---- FomcStepNodeCurve ----------------------------------------------------

    FomcStepNodeCurve::FomcStepNodeCurve(const std::vector<Date>& dates,
                                         const DayCounter& dayCounter)
    : InterpolatedDiscountCurve<LogLinear>(
          dates, std::vector<DiscountFactor>(dates.size(), 1.0), dayCounter) {
        enableExtrapolation();
    }

    void FomcStepNodeCurve::setDiscounts(const std::vector<DiscountFactor>& discounts) {
        QL_REQUIRE(discounts.size() == this->data_.size(),
                   "FomcStepNodeCurve: " << discounts.size() << " discounts for "
                                         << this->data_.size() << " nodes");
        QL_REQUIRE(discounts.front() == 1.0, "FomcStepNodeCurve: first discount must be 1");
        std::copy(discounts.begin(), discounts.end(), this->data_.begin());
        this->interpolation_.update();
        notifyObservers();
    }

    // ---- FomcStepParametrisation ---------------------------------------------

    FomcStepParametrisation::FomcStepParametrisation(const Date& referenceDate,
                                                     std::vector<Date> meetingDecisions,
                                                     Calendar calendar,
                                                     FomcStepCurveConfig config,
                                                     bool withEffr)
    : reference_(referenceDate), calendar_(std::move(calendar)), config_(std::move(config)),
      withEffr_(withEffr) {
        const FomcStepCurveConfig& cfg = config_;
        QL_REQUIRE(cfg.turnPriorMeanBp.size() == 3 && cfg.turnPriorSdBp.size() == 3,
                   "FomcStepParametrisation: three turn priors (ME, QE, YE) expected");

        QL_REQUIRE(static_cast<Real>(Date::maxDate() - reference_) > cfg.curveYears * 365.25,
                   "FomcStepParametrisation: a " << cfg.curveYears << "y curve from " << reference_
                                                 << " runs past QuantLib's last date "
                                                 << Date::maxDate());
        Date end = reference_ + static_cast<Integer>(cfg.curveYears * 365.25);
        for (Date d = reference_; d <= end; ++d)
            days_.push_back(d);
        for (const Date& d : days_)
            if (calendar_.isBusinessDay(d))
                bdays_.push_back(d);
        {
            std::map<Date, Size> bidx;
            for (Size i = 0; i < bdays_.size(); ++i)
                bidx[bdays_[i]] = i;
            Size last = 0;
            for (const Date& d : days_) {
                auto it = bidx.find(d);
                if (it != bidx.end())
                    last = it->second;
                calToB_.push_back(last);
            }
        }

        // A decision moves the next business day's fixing.
        Date horizon = reference_ + static_cast<Integer>(cfg.horizonYears * 365.25);
        std::set<Date> uniqueDecisions(meetingDecisions.begin(), meetingDecisions.end());
        for (const Date& dec : uniqueDecisions) {
            Date eff = calendar_.advance(dec, 1, Days);
            if (reference_ < eff && eff <= horizon) {
                decisions_.push_back(dec);
                effective_.push_back(eff);
            }
        }
        QL_REQUIRE(!decisions_.empty(), "no FOMC meeting effective in (" << reference_ << ", "
                                                                           << horizon
                                                                           << "]; the step region "
                                                                              "would be empty");
        switch_ = effective_.back();
        Real ts = (switch_ - reference_) / 365.0;

        std::vector<Real> knots = {ts};
        for (Real k : cfg.splineKnotsYears)
            if (ts + 0.5 < k && k < static_cast<Real>(cfg.curveYears))
                knots.push_back(k);
        knots.push_back(static_cast<Real>(cfg.curveYears));
        std::vector<Real> t(3, knots.front());
        t.insert(t.end(), knots.begin(), knots.end());
        t.insert(t.end(), 3, knots.back());
        Size nSpline = t.size() - 4;

        // Spread knots: today, the first of each month, the switch date.
        std::vector<Date> knotDates = {reference_};
        for (Date m = firstOfNextMonth(reference_); m < switch_; m = firstOfNextMonth(m))
            knotDates.push_back(m);
        knotDates.push_back(switch_);
        spreadKnots_ = withEffr_ ? knotDates :
                                   std::vector<Date>(knotDates.begin() + 1, knotDates.end());

        Size nMoves = decisions_.size();
        moves_ = {1, 1 + nMoves};
        spread_ = {moves_.second, moves_.second + spreadKnots_.size()};
        turns_ = {spread_.second, spread_.second + 3};
        spline_ = {turns_.second, turns_.second + nSpline - 1};
        names_.emplace_back("level");
        for (const Date& d : decisions_)
            names_.push_back("fomc:" + isoDate(d));
        for (const Date& d : spreadKnots_)
            names_.push_back("spread:" + isoDate(d));
        for (const char* k : {"ME", "QE", "YE"})
            names_.push_back(std::string("turn:") + k);
        for (Size j = 1; j < nSpline; ++j)
            names_.push_back("spline:" + std::to_string(j));

        // Rate rows.
        Size nb = bdays_.size();
        std::vector<std::map<Size, Real> > sofr(nb), effr(nb);
        std::vector<Real> kd;
        for (const Date& d : knotDates)
            kd.push_back(static_cast<Real>(d - reference_));
        Size sLast = spread_.second - 1;
        for (Size i = 0; i < nb; ++i) {
            const Date& b = bdays_[i];
            Real tb = (b - reference_) / 365.0;
            if (tb < ts) {
                sofr[i][0] += 1.0;
                effr[i][0] += 1.0;
                for (Size k = 0; k < nMoves; ++k)
                    if (b >= effective_[k]) {
                        sofr[i][moves_.first + k] += 1.0;
                        effr[i][moves_.first + k] += 1.0;
                    }
                // Spread: hat functions between knots (day counts).
                Real bd = static_cast<Real>(b - reference_);
                Size j = std::upper_bound(kd.begin(), kd.end(), bd) - kd.begin();
                j = (j == 0) ? 0 : j - 1;
                std::vector<std::pair<Size, Real> > w;
                if (j + 1 < kd.size()) {
                    Real theta = (bd - kd[j]) / (kd[j + 1] - kd[j]);
                    w = {{j, 1.0 - theta}, {j + 1, theta}};
                } else {
                    w = {{kd.size() - 1, 1.0}};
                }
                for (const auto& kw : w) {
                    if (!withEffr_ && kw.first == 0)
                        continue;  // spread at today fixed at zero
                    Size col = spread_.first + kw.first - (withEffr_ ? 0 : 1);
                    sofr[i][col] += kw.second;
                }
            } else {
                Real tc = std::min(std::max(tb, ts), static_cast<Real>(cfg.curveYears) - 1e-9);
                std::vector<Real> B = bsplineBasis(t, 3, tc);
                for (auto* rows : {&sofr, &effr}) {
                    auto& row = (*rows)[i];
                    row[0] += B[0];
                    for (Size k = 0; k < nMoves; ++k)
                        row[moves_.first + k] += B[0];
                    row[sLast] += B[0];
                    for (Size j = 1; j < nSpline; ++j)
                        row[spline_.first + j - 1] = B[j];
                }
                // EFFR beyond the switch: the SOFR tail less the final spread.
                effr[i][sLast] -= 1.0;
            }
        }
        // Turns only inside the step region.
        for (Size i = 0; i + 1 < nb; ++i) {
            const Date& b = bdays_[i];
            if (b < switch_ && bdays_[i + 1].month() != b.month()) {
                Size kind = b.month() == December ? 2 :
                            (b.month() == March || b.month() == June || b.month() == September) ?
                                                    1 :
                                                    0;
                for (Size off = 0; off < cfg.turnShape.size(); ++off)
                    if (i + off < nb)
                        sofr[i + off][turns_.first + kind] += cfg.turnShape[off];
            }
        }
        auto toRows = [](const std::vector<std::map<Size, Real> >& maps) {
            std::vector<Row> rows(maps.size());
            for (Size i = 0; i < maps.size(); ++i)
                for (const auto& cv : maps[i])
                    if (cv.second != 0.0)
                        rows[i].emplace_back(cv.first, cv.second);
            return rows;
        };
        sofrRows_ = toRows(sofr);
        effrRows_ = toRows(effr);

        // Fixed jumps: constants on the SOFR path inside the step region.
        QL_REQUIRE(cfg.jumpDates.size() == cfg.jumpBp.size(),
                   "FomcStepCurveConfig: " << cfg.jumpDates.size() << " jump dates for "
                                           << cfg.jumpBp.size() << " jump sizes");
        sofrJump_.assign(bdays_.size(), 0.0);
        for (Size j = 0; j < cfg.jumpDates.size(); ++j) {
            const Date& d = cfg.jumpDates[j];
            if (d < reference_ || d >= switch_)
                continue;  // already past (a rolled curve), or beyond the step region
            auto it = std::lower_bound(bdays_.begin(), bdays_.end(), d);
            QL_REQUIRE(it != bdays_.end() && *it == d,
                       "FomcStepCurveConfig: jump date " << d << " is not a business day of the curve");
            sofrJump_[it - bdays_.begin()] += cfg.jumpBp[j] / 1e4;
        }

        // int S''^2 on the spline as its square-root factor R (R'R = Gram).
        const Size nq = 4000;
        Real qa = ts, qb = static_cast<Real>(cfg.curveYears) - 1e-9;
        Real dt = (qb - qa) / (nq - 1);
        Matrix gram(nSpline, nSpline, 0.0);
        for (Size q = 0; q < nq; ++q) {
            std::vector<Real> d2 = cubicSecondDerivatives(t, qa + q * dt);
            for (Size a = 0; a < nSpline; ++a)
                for (Size b = 0; b < nSpline; ++b)
                    gram[a][b] += d2[a] * d2[b] * dt;
        }
        Matrix curvatureFactor = gramFactor(gram);

        // int S'^2 over the tension region of the tail.
        Real ta = std::max(ts, cfg.tailTensionStartYears);
        Matrix gram1(nSpline, nSpline, 0.0);
        if (cfg.tailTensionLambda > 0.0 && ta < qb) {
            Real dt1 = (qb - ta) / (nq - 1);
            for (Size q = 0; q < nq; ++q) {
                std::vector<Real> d1 = cubicFirstDerivatives(t, ta + q * dt1);
                for (Size a = 0; a < nSpline; ++a)
                    for (Size b = 0; b < nSpline; ++b)
                        gram1[a][b] += d1[a] * d1[b] * dt1;
            }
        }
        tensionFactor_ = gramFactor(gram1);

        // Curve nodes: every calendar day whose rate row differs from the day
        // before inside the step region (log-linear discounts are then exact),
        // every tailNodeDays beyond it.
        Size iSwitch = static_cast<Size>(switch_ - reference_);
        std::set<Size> nodes = {0};
        for (Size i = 1; i <= iSwitch && i < days_.size(); ++i)
            if (sofrRows_[calToB_[i]] != sofrRows_[calToB_[i - 1]] ||
                sofrJump_[calToB_[i]] != sofrJump_[calToB_[i - 1]])
                nodes.insert(i);
        for (Size i = iSwitch; i < days_.size(); i += cfg.tailNodeDays)
            nodes.insert(i);
        nodes.insert(days_.size() - 1);
        nodeIndex_.assign(nodes.begin(), nodes.end());
        for (Size i : nodeIndex_)
            nodeDates_.push_back(days_[i]);

        buildPenalties(curvatureFactor, nSpline);
    }

    void FomcStepParametrisation::buildPenalties(const Matrix& R, Size nSpline) {
        const FomcStepCurveConfig& cfg = config_;
        Size n = size();
        Size nMoves = moves_.second - moves_.first;
        Size nKnots = (spread_.second - spread_.first) + (withEffr_ ? 0 : 1);
        Size rows = (nMoves - 1) + (nKnots - 1) + (nKnots - 2) + 3 + 2 * nSpline + 1;
        penalty_ = Matrix(rows, n, 0.0);
        penaltyOffset_ = Array(rows, 0.0);
        Size r = 0;
        for (Size k = 0; k + 1 < nMoves; ++k, ++r) {
            penalty_[r][moves_.first + k + 1] = 1.0 / cfg.pathSigmaBp;
            penalty_[r][moves_.first + k] = -1.0 / cfg.pathSigmaBp;
        }
        // Spread knot j as a parameter column, or none for the fixed zero.
        auto knotCol = [&](Size j) -> Integer {
            if (!withEffr_ && j == 0)
                return -1;
            return static_cast<Integer>(spread_.first + j - (withEffr_ ? 0 : 1));
        };
        for (Size j = 0; j + 1 < nKnots; ++j, ++r) {
            if (knotCol(j + 1) >= 0)
                penalty_[r][knotCol(j + 1)] += 1.0 / cfg.spreadStepBp;
            if (knotCol(j) >= 0)
                penalty_[r][knotCol(j)] -= 1.0 / cfg.spreadStepBp;
        }
        for (Size j = 0; j + 2 < nKnots; ++j, ++r) {
            if (knotCol(j + 2) >= 0)
                penalty_[r][knotCol(j + 2)] += 1.0 / cfg.spreadCurveBp;
            if (knotCol(j + 1) >= 0)
                penalty_[r][knotCol(j + 1)] -= 2.0 / cfg.spreadCurveBp;
            if (knotCol(j) >= 0)
                penalty_[r][knotCol(j)] += 1.0 / cfg.spreadCurveBp;
        }
        for (Size k = 0; k < 3; ++k, ++r) {
            penalty_[r][turns_.first + k] = 1.0 / cfg.turnPriorSdBp[k];
            penaltyOffset_[r] = -cfg.turnPriorMeanBp[k] / cfg.turnPriorSdBp[k];
        }
        // c = (L + sum D + last spread knot, spline coefficients).
        Real s = std::sqrt(cfg.curvatureLambda);
        Size sLast = spread_.second - 1;
        for (Size a = 0; a < nSpline; ++a, ++r) {
            Real c0 = s * R[a][0];
            penalty_[r][0] += c0;
            for (Size k = moves_.first; k < moves_.second; ++k)
                penalty_[r][k] += c0;
            penalty_[r][sLast] += c0;
            for (Size j = 1; j < nSpline; ++j)
                penalty_[r][spline_.first + j - 1] += s * R[a][j];
        }
        Real st = std::sqrt(cfg.tailTensionLambda);
        for (Size a = 0; a < nSpline; ++a, ++r) {
            Real c0 = st * tensionFactor_[a][0];
            penalty_[r][0] += c0;
            for (Size k = moves_.first; k < moves_.second; ++k)
                penalty_[r][k] += c0;
            penalty_[r][sLast] += c0;
            for (Size j = 1; j < nSpline; ++j)
                penalty_[r][spline_.first + j - 1] += st * tensionFactor_[a][j];
        }
        // The clamped spline equals its last coefficient at the curve's end:
        // pin that forward to the ultimate forward.
        QL_REQUIRE(nSpline > 1, "FomcStepParametrisation: the tail spline needs at least two coefficients");
        penalty_[r][spline_.second - 1] = 1.0 / cfg.ultimateSigmaBp;
        penaltyOffset_[r] = -cfg.ultimateForwardBp / cfg.ultimateSigmaBp;
        ++r;
        QL_ENSURE(r == rows, "FomcStepParametrisation: penalty rows " << r << " != " << rows);
    }

    std::vector<Real> FomcStepParametrisation::rowRates(const std::vector<Row>& rows,
                                                        const Array& x) const {
        QL_REQUIRE(x.size() == size(),
                   "FomcStepParametrisation: " << x.size() << " parameters for " << size());
        std::vector<Real> r(rows.size(), 0.0);
        for (Size i = 0; i < rows.size(); ++i) {
            Real v = 0.0;
            for (const auto& cw : rows[i])
                v += cw.second * x[cw.first];
            r[i] = v / 1e4;
        }
        return r;
    }

    std::vector<Real> FomcStepParametrisation::rates(const Array& x, bool effr) const {
        std::vector<Real> r = rowRates(effr ? effrRows_ : sofrRows_, x);
        if (!effr)
            for (Size i = 0; i < r.size(); ++i)
                r[i] += sofrJump_[i];
        return r;
    }

    std::vector<DiscountFactor> FomcStepParametrisation::nodeDiscounts(const Array& x,
                                                                       bool effr) const {
        std::vector<Real> r = rates(x, effr);
        std::vector<DiscountFactor> out;
        out.reserve(nodeIndex_.size());
        Real logDf = 0.0;
        Size next = 0;
        for (Size i = 0; i < days_.size() && next < nodeIndex_.size(); ++i) {
            if (nodeIndex_[next] == i) {
                out.push_back(std::exp(logDf));
                ++next;
            }
            logDf -= std::log1p(r[calToB_[i]] / 360.0);
        }
        return out;
    }

    Matrix FomcStepParametrisation::dLogDiscountDx(const Array& x) const {
        std::vector<Real> r = rates(x, false);
        Size n = size();
        Matrix out(nodeIndex_.size(), n, 0.0);
        std::vector<Real> cum(n, 0.0);
        Size next = 0;
        for (Size i = 0; i < days_.size() && next < nodeIndex_.size(); ++i) {
            if (nodeIndex_[next] == i) {
                for (Size j = 0; j < n; ++j)
                    out[next][j] = -cum[j];
                ++next;
            }
            Real rb = r[calToB_[i]];
            Real w = 1.0 / (360.0 * (1.0 + rb / 360.0)) / 1e4;
            for (const auto& cw : sofrRows_[calToB_[i]])
                cum[cw.first] += w * cw.second;
        }
        return out;
    }

    Array FomcStepParametrisation::initialGuess(Real overnightBp) const {
        Array x(size(), 0.0);
        x[0] = overnightBp;
        for (Size k = 0; k < 3; ++k)
            x[turns_.first + k] = config_.turnPriorMeanBp[k];
        for (Size j = spline_.first; j < spline_.second; ++j)
            x[j] = overnightBp;
        return x;
    }

    // ---- FomcStepFit ----------------------------------------------------------

    FomcStepFit::FomcStepFit(ext::shared_ptr<FomcStepParametrisation> parametrisation,
                             std::vector<ext::shared_ptr<RateHelper> > helpers,
                             std::vector<std::string> names,
                             std::vector<Real> bpValues,
                             std::vector<Real> halfWidthsBp,
                             std::vector<bool> onEffrCurve,
                             std::vector<bool> synthetic,
                             const DayCounter& dayCounter)
    : param_(std::move(parametrisation)), helpers_(std::move(helpers)), names_(std::move(names)),
      bpValues_(std::move(bpValues)), halfWidthsBp_(std::move(halfWidthsBp)),
      onEffr_(std::move(onEffrCurve)), synthetic_(std::move(synthetic)) {
        Size n = helpers_.size();
        QL_REQUIRE(n > 0, "FomcStepFit: no targets");
        QL_REQUIRE(names_.size() == n && bpValues_.size() == n && halfWidthsBp_.size() == n &&
                       onEffr_.size() == n && synthetic_.size() == n,
                   "FomcStepFit: every target needs a name, bp value, band, curve and flag");
        sofr_ = ext::make_shared<FomcStepNodeCurve>(param_->nodeDates(), dayCounter);
        effr_ = ext::make_shared<FomcStepNodeCurve>(param_->nodeDates(), dayCounter);
        syntheticQuotes_.resize(n);
        for (Size i = 0; i < n; ++i) {
            QL_REQUIRE(halfWidthsBp_[i] > 0.0,
                       "FomcStepFit: target " << names_[i] << " has no positive band");
            QL_REQUIRE(!onEffr_[i] || param_->withEffr(),
                       "FomcStepFit: target " << names_[i]
                                              << " is on the EFFR curve but the fit has no EFFR leg");
            helpers_[i]->setTermStructure(onEffr_[i] ? effr_.get() : sofr_.get());
            if (synthetic_[i]) {
                syntheticQuotes_[i] =
                    ext::dynamic_pointer_cast<SimpleQuote>(helpers_[i]->quote().currentLink());
                QL_REQUIRE(syntheticQuotes_[i],
                           "FomcStepFit: synthetic target " << names_[i]
                                                            << " needs a SimpleQuote");
            }
        }
    }

    void FomcStepFit::show(const Array& x) {
        sofr_->setDiscounts(param_->nodeDiscounts(x, false));
        if (param_->withEffr())
            effr_->setDiscounts(param_->nodeDiscounts(x, true));
        shown_ = x;
    }

    Array FomcStepFit::quoteErrorsBp(const Array& x) {
        show(x);
        Array e(helpers_.size());
        for (Size i = 0; i < helpers_.size(); ++i)
            e[i] = (helpers_[i]->impliedQuote() - helpers_[i]->quote()->value()) / bpValues_[i];
        return e;
    }

    Matrix FomcStepFit::jacobianAt(const Array& x, const Array& e) {
        const Real h = 1e-4;  // bp
        Matrix J(e.size(), x.size());
        for (Size j = 0; j < x.size(); ++j) {
            Array xj = x;
            xj[j] += h;
            Array ej = quoteErrorsBp(xj);
            for (Size i = 0; i < e.size(); ++i)
                J[i][j] = (ej[i] - e[i]) / h;
        }
        return J;
    }

    Array FomcStepFit::solveLinear(
        const Array& x, const Array& e, const Matrix& Je, const Array& hw, bool band) const {
        const FomcStepCurveConfig& cfg = param_->config();
        const Matrix& P = param_->penaltyMatrix();
        const Array& p0 = param_->penaltyOffset();
        Size m = e.size();
        Array pen = P * x + p0;

        Matrix Aq(m, Je.columns());
        Array bq(m);
        for (Size i = 0; i < m; ++i) {
            for (Size j = 0; j < Je.columns(); ++j)
                Aq[i][j] = Je[i][j] / hw[i];
            bq[i] = -e[i] / hw[i];
        }
        Array d = leastSquares(stack(Aq, P), concat(bq, -pen));
        if (!band)
            return d;

        // Convex, piecewise quadratic: fix which quotes sit past their band
        // (and on which side), solve that quadratic exactly, repeat.
        Array inner = hw * cfg.bandInnerK;
        std::set<std::vector<int> > seen;
        for (Size iter = 0; iter < 50; ++iter) {
            Array ee = e + Je * d;
            std::vector<int> sgn(m, 0);
            for (Size i = 0; i < m; ++i)
                if (std::fabs(ee[i]) > hw[i])
                    sgn[i] = ee[i] > 0.0 ? 1 : -1;
            if (!seen.insert(sgn).second)
                break;  // the set cycles
            Size nOut = 0;
            for (int s : sgn)
                nOut += (s != 0);
            Matrix A(m + nOut, Je.columns());
            Array b(m + nOut);
            Size r = 0;
            for (Size i = 0; i < m; ++i, ++r) {
                for (Size j = 0; j < Je.columns(); ++j)
                    A[r][j] = Je[i][j] / inner[i];
                b[r] = -e[i] / inner[i];
            }
            for (Size i = 0; i < m; ++i) {
                if (sgn[i] == 0)
                    continue;
                for (Size j = 0; j < Je.columns(); ++j)
                    A[r][j] = Je[i][j] / cfg.bandWallBp;
                b[r] = -(e[i] - sgn[i] * hw[i]) / cfg.bandWallBp;
                ++r;
            }
            Array dNew = leastSquares(stack(A, P), concat(b, -pen));
            Array eNew = e + Je * dNew;
            bool settled = true;
            for (Size i = 0; i < m && settled; ++i) {
                int s = std::fabs(eNew[i]) > hw[i] ? (eNew[i] > 0.0 ? 1 : -1) : 0;
                settled = (s == sgn[i]);
            }
            if (settled)
                return dNew;
            d = dNew;
        }
        // The active set cycles between quotes on their band edges: finish
        // this convex step with Levenberg-Marquardt from here.
        LinearBandCost cost(e, Je, hw, inner, cfg.bandWallBp, P, p0, x);
        NoConstraint constraint;
        Problem problem(cost, constraint, d);
        LevenbergMarquardt lm;
        lm.minimize(problem, EndCriteria(500, 100, 1e-12, 1e-12, 1e-12));
        return problem.currentValue();
    }

    void FomcStepFit::fit(const Array& x0,
                          const std::string& loss,
                          bool holdSynthetic,
                          bool reuseJacobian) {
        QL_REQUIRE(loss == "band" || loss == "quadratic",
                   "FomcStepFit: loss must be 'band' or 'quadratic', got " << loss);
        const FomcStepCurveConfig& cfg = param_->config();
        Size nT = helpers_.size(), n = param_->size();
        QL_REQUIRE(x0.size() == n, "FomcStepFit: " << x0.size() << " start values for " << n);
        Array hw(halfWidthsBp_.begin(), halfWidthsBp_.end());
        std::vector<Size> real;
        for (Size i = 0; i < nT; ++i)
            if (holdSynthetic || !synthetic_[i])
                real.push_back(i);
        Array hwReal(real.size());
        for (Size k = 0; k < real.size(); ++k)
            hwReal[k] = hw[real[k]];

        Array x = x0;
        Array e = quoteErrorsBp(x);
        Matrix Je;
        bool haveJacobian = reuseJacobian && jacobian_.rows() == nT && jacobian_.columns() == n;
        if (haveJacobian)
            Je = jacobian_;
        Real departure = 0.0;
        bool converged = false;
        for (outerSteps_ = 1; outerSteps_ <= cfg.maxOuterSteps; ++outerSteps_) {
            if (!haveJacobian) {
                Je = jacobianAt(x, e);
                haveJacobian = true;
            }
            Array eReal(real.size());
            Matrix JReal(real.size(), n);
            for (Size k = 0; k < real.size(); ++k) {
                eReal[k] = e[real[k]];
                std::copy(Je.row_begin(real[k]), Je.row_end(real[k]), JReal.row_begin(k));
            }
            Array d = solveLinear(x, eReal, JReal, hwReal, loss == "band");
            Array predicted = eReal + JReal * d;
            x += d;
            e = quoteErrorsBp(x);  // also shows x
            departure = 0.0;
            for (Size k = 0; k < real.size(); ++k)
                departure = std::max(departure, std::fabs(e[real[k]] - predicted[k]));
            if (departure < cfg.nonlinearTolBp) {
                converged = true;
                break;
            }
            haveJacobian = false;
        }
        QL_REQUIRE(converged, "FOMC-step fit did not converge in "
                                  << cfg.maxOuterSteps << " steps (last re-pricing off the "
                                  << "linear model by " << departure << " bp)");
        finish(x, e, Je, loss, holdSynthetic);
    }

    void FomcStepFit::restore(const Array& x,
                              const Matrix& jacobian,
                              const std::string& loss,
                              bool holdSynthetic,
                              Size outerSteps) {
        QL_REQUIRE(loss == "band" || loss == "quadratic",
                   "FomcStepFit: loss must be 'band' or 'quadratic', got " << loss);
        Size nT = helpers_.size(), n = param_->size();
        QL_REQUIRE(x.size() == n, "FomcStepFit: " << x.size() << " parameters for " << n);
        QL_REQUIRE(jacobian.rows() == nT && jacobian.columns() == n,
                   "FomcStepFit: a " << jacobian.rows() << "x" << jacobian.columns()
                                     << " quote Jacobian for " << nT << " targets and " << n
                                     << " parameters");
        // The quote errors at x, as the fit's last step measured them; the
        // rest is the fit's own post-solve code on the stored Jacobian.
        Array e = quoteErrorsBp(x);
        outerSteps_ = outerSteps;
        finish(x, e, jacobian, loss, holdSynthetic);
    }

    void FomcStepFit::finish(const Array& x,
                             Array e,
                             const Matrix& Je,
                             const std::string& loss,
                             bool holdSynthetic) {
        const FomcStepCurveConfig& cfg = param_->config();
        Size nT = helpers_.size(), n = param_->size();
        Array hw(halfWidthsBp_.begin(), halfWidthsBp_.end());

        // Synthetic benchmarks take the fitted value: zero error, curve unchanged.
        if (!holdSynthetic)
            for (Size i = 0; i < nT; ++i)
                if (synthetic_[i]) {
                    syntheticQuotes_[i]->setValue(helpers_[i]->impliedQuote());
                    e[i] = 0.0;
                }

        // Risk: the band-weighted quadratic model at the solution.
        const Matrix& P = param_->penaltyMatrix();
        Matrix Jw(nT, n);
        for (Size i = 0; i < nT; ++i)
            for (Size j = 0; j < n; ++j)
                Jw[i][j] = Je[i][j] / hw[i];
        SVD svd(stack(Jw, P));
        const Array& s = svd.singularValues();
        const Matrix& V = svd.V();
        Real sMax = *std::max_element(s.begin(), s.end());
        covariance_ = Matrix(n, n, 0.0);
        for (Size k = 0; k < s.size(); ++k) {
            if (s[k] * s[k] <= 1e-15 * sMax * sMax)
                continue;
            Real inv = 1.0 / (s[k] * s[k]);
            for (Size a = 0; a < n; ++a)
                for (Size b = 0; b < n; ++b)
                    covariance_[a][b] += V[a][k] * V[b][k] * inv;
        }
        // A market move of dq bp shifts quote residual i by -dq/hw_i.
        Matrix JeT(n, nT);
        for (Size i = 0; i < nT; ++i)
            for (Size j = 0; j < n; ++j)
                JeT[j][i] = Je[i][j] / (hw[i] * hw[i]);
        dxdq_ = covariance_ * JeT;
        dLogDfDq_ = param_->dLogDiscountDx(x) * dxdq_;
        jacobian_ = Je;
        x_ = x;
        residualBp_ = e;

        Array pen = P * x + param_->penaltyOffset();
        Real c = DotProduct(pen, pen);
        for (Size i = 0; i < nT; ++i) {
            if (loss == "band") {
                Real a = e[i] / (hw[i] * cfg.bandInnerK);
                Real b = std::max(std::fabs(e[i]) - hw[i], 0.0) / cfg.bandWallBp;
                c += a * a + b * b;
            } else {
                c += (e[i] / hw[i]) * (e[i] / hw[i]);
            }
        }
        cost_ = 0.5 * c;
    }

    void FomcStepFit::setHalfWidthsBp(const std::vector<Real>& halfWidthsBp) {
        QL_REQUIRE(halfWidthsBp.size() == helpers_.size(),
                   "FomcStepFit: " << halfWidthsBp.size() << " half-widths for "
                                   << helpers_.size() << " targets");
        for (Size i = 0; i < halfWidthsBp.size(); ++i)
            QL_REQUIRE(halfWidthsBp[i] > 0.0, "FomcStepFit: half-width of target "
                                                  << names_[i] << " must be positive, got "
                                                  << halfWidthsBp[i]);
        halfWidthsBp_ = halfWidthsBp;
    }

    std::vector<std::string> FomcStepFit::outsideBand() const {
        std::vector<std::string> out;
        for (Size i = 0; i < residualBp_.size(); ++i)
            if (std::fabs(residualBp_[i]) > halfWidthsBp_[i])
                out.push_back(names_[i]);
        return out;
    }

}
