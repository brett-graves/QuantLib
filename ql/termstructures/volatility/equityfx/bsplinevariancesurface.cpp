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

#include <ql/patterns/visitor.hpp>
#include <ql/termstructures/volatility/equityfx/bsplinevariancesurface.hpp>
#include <ql/termstructures/volatility/equityfx/puredividend.hpp>
#include <ql/utilities/null.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace QuantLib {

    namespace {

        constexpr int maxDegree = 7;
        constexpr Real leeBound = 2.0;

        /* Value and first two derivatives of the clamped B-spline with knots U
           and coefficients c at x in [U[p], U[n]] (Piegl & Tiller, The NURBS
           Book, algorithms A2.1 and A2.3).  Stack arrays only: this runs inside
           pricing loops. */
        std::array<Real, 3> deBoor(const std::vector<Real>& U,
                                   const std::vector<Real>& c,
                                   int p,
                                   Real x) {
            const int n = static_cast<int>(c.size());
            int span;
            if (x >= U[n]) {
                span = n - 1;
            } else {
                span = static_cast<int>(std::upper_bound(U.begin() + p, U.begin() + n + 1, x) -
                                        U.begin()) - 1;
                span = std::max(p, std::min(span, n - 1));
            }
            constexpr int nd = 2;
            Real ndu[maxDegree + 1][maxDegree + 1];
            Real left[maxDegree + 1], right[maxDegree + 1];
            ndu[0][0] = 1.0;
            for (int j = 1; j <= p; ++j) {
                left[j] = x - U[span + 1 - j];
                right[j] = U[span + j] - x;
                Real saved = 0.0;
                for (int r = 0; r < j; ++r) {
                    ndu[j][r] = right[r + 1] + left[j - r];
                    const Real temp = ndu[r][j - 1] / ndu[j][r];
                    ndu[r][j] = saved + right[r + 1] * temp;
                    saved = left[j - r] * temp;
                }
                ndu[j][j] = saved;
            }
            Real ders[nd + 1][maxDegree + 1];
            for (int j = 0; j <= p; ++j)
                ders[0][j] = ndu[j][p];
            Real a[2][maxDegree + 1];
            for (int r = 0; r <= p; ++r) {
                int s1 = 0, s2 = 1;
                a[0][0] = 1.0;
                for (int k = 1; k <= nd; ++k) {
                    Real d = 0.0;
                    const int rk = r - k, pk = p - k;
                    if (r >= k) {
                        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                        d = a[s2][0] * ndu[rk][pk];
                    }
                    const int j1 = rk >= -1 ? 1 : -rk;
                    const int j2 = (r - 1 <= pk) ? k - 1 : p - r;
                    for (int j = j1; j <= j2; ++j) {
                        a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                        d += a[s2][j] * ndu[rk + j][pk];
                    }
                    if (r <= pk) {
                        a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                        d += a[s2][k] * ndu[r][pk];
                    }
                    ders[k][r] = d;
                    std::swap(s1, s2);
                }
            }
            Real factor = p;
            for (int k = 1; k <= nd; ++k) {
                for (int j = 0; j <= p; ++j)
                    ders[k][j] *= factor;
                factor *= (p - k);
            }
            std::array<Real, 3> out = {0.0, 0.0, 0.0};
            for (int k = 0; k <= nd; ++k)
                for (int j = 0; j <= p; ++j)
                    out[k] += ders[k][j] * c[span - p + j];
            return out;
        }

    }

    BSplineVarianceSurface::BSplineVarianceSurface(
        const Date& referenceDate,
        const std::vector<Date>& dates,
        const std::vector<std::vector<Real>>& knots,
        const std::vector<std::vector<Real>>& coefficients,
        Natural degree,
        Handle<Quote> spot,
        Handle<YieldTermStructure> riskFreeRate,
        Handle<YieldTermStructure> dividendYield,
        const DayCounter& dc)
    : BSplineVarianceSurface(referenceDate, dates, knots, coefficients, degree, std::move(spot),
                             std::move(riskFreeRate), std::move(dividendYield),
                             DividendSchedule(), dc) {}

    BSplineVarianceSurface::BSplineVarianceSurface(
        const Date& referenceDate,
        const std::vector<Date>& dates,
        const std::vector<std::vector<Real>>& knots,
        const std::vector<std::vector<Real>>& coefficients,
        Natural degree,
        Handle<Quote> spot,
        Handle<YieldTermStructure> riskFreeRate,
        Handle<YieldTermStructure> dividendYield,
        DividendSchedule dividends,
        const DayCounter& dc)
    : BlackVolatilityTermStructure(referenceDate, Calendar(), Following, dc), degree_(degree),
      spot_(std::move(spot)), riskFreeRate_(std::move(riskFreeRate)),
      dividendYield_(std::move(dividendYield)), dividends_(std::move(dividends)) {
        QL_REQUIRE(!spot_.empty() && !riskFreeRate_.empty() && !dividendYield_.empty(),
                   "spot/rate/dividend handles must not be empty");
        QL_REQUIRE(degree_ >= 2 && degree_ <= static_cast<Natural>(maxDegree),
                   "degree must be in [2, " << maxDegree << "], got " << degree_);
        registerWith(spot_);
        registerWith(riskFreeRate_);
        registerWith(dividendYield_);
        initialize(dates, knots, coefficients);
    }

    void BSplineVarianceSurface::initialize(const std::vector<Date>& dates,
                                            const std::vector<std::vector<Real>>& knots,
                                            const std::vector<std::vector<Real>>& coefficients) {
        const Size N = dates.size();
        QL_REQUIRE(N > 0, "at least one pillar is required");
        QL_REQUIRE(knots.size() == N && coefficients.size() == N,
                   "dates (" << N << "), knots (" << knots.size() << ") and coefficients ("
                             << coefficients.size() << ") sizes differ");

        std::vector<Size> order(N);
        std::iota(order.begin(), order.end(), Size(0));
        std::sort(order.begin(), order.end(),
                  [&dates](Size a, Size b) { return dates[a] < dates[b]; });

        const Size p = degree_;
        times_.resize(N);
        slices_.resize(N);
        dates_.resize(N);
        atm_.resize(N);
        for (Size i = 0; i < N; ++i) {
            const Size src = order[i];
            const Time T = dayCounter().yearFraction(referenceDate(), dates[src]);
            QL_REQUIRE(T > 0.0, "pillar " << dates[src] << " is not after the reference date");
            QL_REQUIRE(i == 0 || T > times_[i - 1],
                       "duplicate or non-increasing pillar at " << dates[src]);
            times_[i] = T;
            dates_[i] = dates[src];

            const std::vector<Real>& U = knots[src];
            const std::vector<Real>& c = coefficients[src];
            const Size n = c.size();
            QL_REQUIRE(n >= p + 1 && U.size() == n + p + 1,
                       "pillar " << dates[src] << ": need >= " << p + 1
                                 << " coefficients and n + degree + 1 knots, got " << n
                                 << " coefficients and " << U.size() << " knots");
            for (Size j = 0; j < U.size(); ++j) {
                QL_REQUIRE(std::isfinite(U[j]), "pillar " << dates[src] << ": knot " << j
                                                          << " is not finite");
                QL_REQUIRE(j == 0 || U[j] >= U[j - 1],
                           "pillar " << dates[src] << ": knots must be non-decreasing");
            }
            for (Size j = 1; j <= p; ++j)
                QL_REQUIRE(U[j] == U[0] && U[n + j - 1] == U[n + p],
                           "pillar " << dates[src] << ": knot vector must be clamped ("
                                     << p + 1 << " equal knots at each end)");
            QL_REQUIRE(U[p] < U[n], "pillar " << dates[src] << ": empty knot range");
            for (Size j = 0; j < n; ++j)
                QL_REQUIRE(std::isfinite(c[j]), "pillar " << dates[src] << ": coefficient " << j
                                                          << " is not finite");

            Slice& s = slices_[i];
            s.knots = U;
            s.coefficients = c;
            s.kL = U[p];
            s.kR = U[n];
            s.left = deBoor(s.knots, s.coefficients, static_cast<int>(p), s.kL);
            s.right = deBoor(s.knots, s.coefficients, static_cast<int>(p), s.kR);
            QL_REQUIRE(s.left[0] > 0.0 && s.right[0] > 0.0,
                       "pillar " << dates[src] << ": total variance at the knot-range ends must be "
                                 "positive, got " << s.left[0] << " and " << s.right[0]);
            QL_REQUIRE(std::fabs(s.left[1]) <= leeBound && std::fabs(s.right[1]) <= leeBound,
                       "pillar " << dates[src] << ": wing slopes dw/dk " << s.left[1] << " and "
                                 << s.right[1] << " exceed Lee's bound " << leeBound);
            atm_[i] = derivatives(i, 0.0)[0];
            QL_REQUIRE(atm_[i] > 0.0, "pillar " << dates[src] << ": ATM total variance "
                                                  << atm_[i] << " is not positive");
        }
    }

    Date BSplineVarianceSurface::maxDate() const { return Date::maxDate(); }

    Real BSplineVarianceSurface::minStrike() const { return 0.0; }

    Real BSplineVarianceSurface::maxStrike() const { return QL_MAX_REAL; }

    Real BSplineVarianceSurface::forward(Time t) const { return forward(t, spot_->value()); }

    Real BSplineVarianceSurface::forward(Time t, Real spot) const {
        // The FD engines' spot dividend model: each cash dividend grows at
        // r - q from its ex-date (gh #511), as ParametricVolTermStructure.
        return cashDividendForward(spot, dividends_, *riskFreeRate_.currentLink(),
                                   *dividendYield_.currentLink(), referenceDate(), dayCounter(),
                                   t);
    }

    Real BSplineVarianceSurface::dividendPV(Time t) const {
        return cashDividendPV(dividends_, *riskFreeRate_.currentLink(),
                              *dividendYield_.currentLink(), referenceDate(), dayCounter(), t);
    }

    void BSplineVarianceSurface::setPureDividendCoordinates(bool pure) {
        if (pure == pureDividend_)
            return;
        pureDividend_ = pure;
        notifyObservers();
    }

    Real BSplineVarianceSurface::coordinate(Time t, Real strike) const {
        const Real f = forward(t);
        const Real d = pureDividend_ ? dividendPV(t) : 0.0;
        if (d == 0.0)
            return std::log(strike / f);
        QL_REQUIRE(f > d, "forward " << f << " at t=" << t << " not above the PV " << d
                                     << " of the dividends still to come");
        // A strike at or below D(t) has no optionality left in the pure model;
        // the floor keeps x finite (pureDividendBlackVol uses the same).
        return std::log(std::max((strike - d) / (f - d), 1e-12));
    }

    std::array<Real, 3> BSplineVarianceSurface::inside(const Slice& s, Real k) const {
        return deBoor(s.knots, s.coefficients, static_cast<int>(degree_), k);
    }

    std::array<Real, 3> BSplineVarianceSurface::derivatives(Size i, Real k) const {
        QL_REQUIRE(i < slices_.size(), "pillar index " << i << " out of range");
        const Slice& s = slices_[i];
        if (k < s.kL)
            return {s.left[0] + s.left[1] * (k - s.kL), s.left[1], 0.0};
        if (k > s.kR)
            return {s.right[0] + s.right[1] * (k - s.kR), s.right[1], 0.0};
        return inside(s, k);
    }

    Real BSplineVarianceSurface::sliceTotalVariance(Size i, Real k) const {
        return derivatives(i, k)[0];
    }

    std::vector<Real> BSplineVarianceSurface::sliceTotalVarianceDerivatives(Size i, Real k) const {
        const std::array<Real, 3> d = derivatives(i, k);
        return {d[0], d[1], d[2]};
    }

    void BSplineVarianceSurface::bracket(Time t, Size& lo, Size& hi, Real& a) const {
        QL_REQUIRE(t > 0.0, "time must be > 0");
        const Size N = times_.size();
        if (t < times_.front() || N == 1) {
            lo = hi = 0;
            a = t / times_.front();
            return;
        }
        if (t >= times_.back()) {
            lo = N - 2;
            hi = N - 1;
        } else {
            // At a pillar date the interval starting there is used (forward-looking).
            hi = static_cast<Size>(std::upper_bound(times_.begin(), times_.end(), t) -
                                   times_.begin());
            lo = hi - 1;
        }
        a = (t - times_[lo]) / (times_[hi] - times_[lo]);
    }

    BSplineVarianceSurface::Bracket BSplineVarianceSurface::bracketAt(Time t) const {
        Bracket b{};
        bracket(t, b.lo, b.hi, b.a);
        b.dt = b.lo == b.hi ? times_.front() : times_[b.hi] - times_[b.lo];
        return b;
    }

    std::array<Real, 4> BSplineVarianceSurface::state(const Bracket& b, Real x) const {
        if (b.lo == b.hi) {
            const std::array<Real, 3> w0 = derivatives(0, x);
            return {b.a * w0[0], b.a * w0[1], b.a * w0[2], w0[0] / b.dt};
        }
        const std::array<Real, 3> wl = derivatives(b.lo, x), wh = derivatives(b.hi, x);
        std::array<Real, 4> s{};
        for (Size j = 0; j < 3; ++j)
            s[j] = (1.0 - b.a) * wl[j] + b.a * wh[j];
        s[3] = (wh[0] - wl[0]) / b.dt;
        return s;
    }

    void BSplineVarianceSurface::setIntradayClock(const Handle<Quote>& now,
                                                  const Handle<Quote>& progress,
                                                  const Date& session,
                                                  Time sessionClose,
                                                  Time fitTime,
                                                  Real fitProgress,
                                                  Real close) {
        QL_REQUIRE(!now.empty() && !progress.empty(),
                   "setIntradayClock: now and progress handles must not be empty");
        QL_REQUIRE(session > referenceDate(), "setIntradayClock: session "
                                                  << session << " is not after the reference date "
                                                  << referenceDate());
        QL_REQUIRE(fitProgress >= 0.0 && fitProgress < 1.0,
                   "setIntradayClock: fit progress " << fitProgress << " outside [0, 1)");
        QL_REQUIRE(fitTime < sessionClose, "setIntradayClock: fit time "
                                               << fitTime << " is not before the session close "
                                               << sessionClose);
        // Pillars on fit-relative time: each date's close less the fit instant.
        for (Size i = 0; i < dates_.size(); ++i) {
            times_[i] = (dates_[i] - referenceDate()) / 365.0 + close - fitTime;
            QL_REQUIRE(times_[i] > 0.0, "setIntradayClock: pillar " << dates_[i]
                                                                    << " is not after the fit");
        }
        const Time r0 = sessionClose - fitTime;
        QL_REQUIRE(times_.front() >= r0 - 1e-12, "setIntradayClock: pillar " << dates_.front()
                                                     << " closes before the session "
                                                     << session);
        if (!now_.empty())
            unregisterWith(now_);
        if (!progress_.empty())
            unregisterWith(progress_);
        now_ = now;
        progress_ = progress;
        fitTime_ = fitTime;
        sessionClose_ = sessionClose;
        fitProgress_ = fitProgress;
        registerWith(now_);
        registerWith(progress_);
        notifyObservers();
    }

    BSplineVarianceSurface::Clock BSplineVarianceSurface::clock() const {
        const Real u = progress_->value();
        QL_REQUIRE(u >= fitProgress_ && u <= 1.0, "day progress " << u << " outside ["
                                                                  << fitProgress_ << ", 1]");
        Clock c{};
        c.rNow = now_->value() - fitTime_;
        c.r0 = sessionClose_ - fitTime_;
        c.v = (u - fitProgress_) / (1.0 - fitProgress_);
        QL_REQUIRE(c.rNow >= -1e-12, "intraday clock: now is before the fit");
        QL_REQUIRE(c.rNow < c.r0 || c.v == 1.0,
                   "intraday clock: the session has closed but its variance is not spent (v = "
                       << c.v << ")");
        return c;
    }

    Time BSplineVarianceSurface::timeToSessionClose() const {
        QL_REQUIRE(hasIntradayClock(), "timeToSessionClose: no intraday clock");
        const Clock c = clock();
        return c.r0 - c.rNow;
    }

    Real BSplineVarianceSurface::sessionVarianceLeft() const {
        QL_REQUIRE(hasIntradayClock(), "sessionVarianceLeft: no intraday clock");
        return 1.0 - clock().v;
    }

    std::array<Real, 4> BSplineVarianceSurface::remaining(Time t, Real x) const {
        if (now_.empty())
            return state(bracketAt(t), x);
        const Clock c = clock();
        const Real left = 1.0 - c.v;
        const Time r = c.rNow + t;
        // Today's close: the fitted surface there, of which 1 - v is left.
        const std::array<Real, 4> today = state(bracketAt(c.r0), x);
        const Size N = times_.size();
        // First pillar after today's close.
        Size first = 0;
        while (first < N && times_[first] <= c.r0 + 1e-12)
            ++first;
        if (r <= c.r0 || first == N) {
            // From now to today's close, linear in calendar time.
            if (left == 0.0)
                return {0.0, 0.0, 0.0, 0.0};
            const Time span = c.r0 - c.rNow;
            const Real f = (r - c.rNow) / span;
            return {left * today[0] * f, left * today[1] * f, left * today[2] * f,
                    left * today[0] / span};
        }
        // Later pillars: each smile scaled so its ATM variance loses v times
        // today's (the front of the term structure slides off).
        const Real atToday = state(bracketAt(c.r0), 0.0)[0];
        auto scale = [&](Size i) { return 1.0 - c.v * atToday / atm_[i]; };
        auto node = [&](Size i) {
            const std::array<Real, 3> w = derivatives(i, x);
            const Real s = scale(i);
            return std::array<Real, 3>{s * w[0], s * w[1], s * w[2]};
        };
        std::array<Real, 3> lo, hi;
        Time rLo, rHi;
        Size hiIdx = first;
        while (hiIdx + 1 < N && times_[hiIdx] < r)
            ++hiIdx;
        if (hiIdx == first) {
            lo = {left * today[0], left * today[1], left * today[2]};
            rLo = c.r0;
        } else {
            lo = node(hiIdx - 1);
            rLo = times_[hiIdx - 1];
        }
        hi = node(hiIdx);
        rHi = times_[hiIdx];
        const Real a = (r - rLo) / (rHi - rLo);
        std::array<Real, 4> out{};
        for (Size j = 0; j < 3; ++j)
            out[j] = (1.0 - a) * lo[j] + a * hi[j];
        out[3] = (hi[0] - lo[0]) / (rHi - rLo);
        return out;
    }

    Real BSplineVarianceSurface::totalVariance(Real k, Time t) const {
        return remaining(t, k)[0];
    }

    Real BSplineVarianceSurface::localVariance(Real k, Time t, Real shift) const {
        const std::array<Real, 4> w = remaining(t, k + shift);
        const Real dwdt = w[3];
        if (w[0] == 0.0 && dwdt == 0.0 && hasIntradayClock())
            return 0.0;  // the session's variance is spent
        QL_REQUIRE(w[0] > 0.0, "non-positive total variance " << w[0] << " at (k=" << k
                                                              << ", t=" << t << ")");
        QL_REQUIRE(dwdt >= 0.0, "calendar arbitrage: dw/dt = " << dwdt << " at (k=" << k
                                                               << ", t=" << t << ")");
        const Real v = 1.0 - 0.5 * k * w[1] / w[0];
        const Real g = v * v - 0.25 * w[1] * w[1] * (1.0 / w[0] + 0.25) + 0.5 * w[2];
        QL_REQUIRE(g > 0.0, "butterfly arbitrage: g = " << g << " at (k=" << k << ", t=" << t
                                                        << ")");
        return dwdt / g;
    }

    Size BSplineVarianceSurface::localVarianceSlice(Time t, const Real* k, Size n,
                                                    Real* out, Real shift) const {
        const bool clocked = hasIntradayClock();
        const Bracket b = clocked ? Bracket{} : bracketAt(t);
        Size nIllegal = 0;
        for (Size j = 0; j < n; ++j) {
            const Real x = k[j] + shift;
            const std::array<Real, 4> w = clocked ? remaining(t, x) : state(b, x);
            const Real dwdt = w[3];
            if (clocked && w[0] == 0.0 && dwdt == 0.0) {
                out[j] = 0.0;
                continue;
            }
            const Real u = 1.0 - 0.5 * k[j] * w[1] / w[0];
            const Real g = u * u - 0.25 * w[1] * w[1] * (1.0 / w[0] + 0.25) + 0.5 * w[2];
            if (!(w[0] > 0.0) || !(dwdt >= 0.0) || !(g > 0.0)) {
                out[j] = Null<Real>();
                ++nIllegal;
            } else {
                out[j] = dwdt / g;
            }
        }
        return nIllegal;
    }

    Volatility BSplineVarianceSurface::blackVolImpl(Time t, Real strike) const {
        if (t < 1e-14)
            t = 1e-14;
        if (pureDividend_) {
            const Real d = dividendPV(t);
            if (d != 0.0) {
                const Real x = coordinate(t, strike);
                const Real wX = totalVariance(x, t);
                QL_REQUIRE(wX >= 0.0, "negative total variance " << wX << " at (x=" << x
                                                                  << ", t=" << t << ")");
                return pureDividendBlackVol(t, strike, forward(t), d, wX);
            }
        }
        const Real k = std::log(strike / forward(t));
        const Real w = totalVariance(k, t);
        QL_REQUIRE(w >= 0.0, "negative total variance " << w << " at (k=" << k << ", t=" << t
                                                         << ")");
        return std::sqrt(w / t);
    }

    void BSplineVarianceSurface::accept(AcyclicVisitor& v) {
        auto* v1 = dynamic_cast<Visitor<BSplineVarianceSurface>*>(&v);
        if (v1 != nullptr)
            v1->visit(*this);
        else
            BlackVolatilityTermStructure::accept(v);
    }

    BSplineLocalVolSurface::BSplineLocalVolSurface(
        ext::shared_ptr<BSplineVarianceSurface> blackSurface)
    : LocalVolTermStructure(blackSurface ? blackSurface->businessDayConvention() : Following,
                            blackSurface ? blackSurface->dayCounter() : DayCounter()),
      blackSurface_(std::move(blackSurface)) {
        QL_REQUIRE(blackSurface_, "blackSurface must not be null");
        registerWith(blackSurface_);
    }

    BSplineLocalVolSurface::BSplineLocalVolSurface(
        ext::shared_ptr<BSplineVarianceSurface> blackSurface, Handle<Quote> diffusionSpot)
    : BSplineLocalVolSurface(std::move(blackSurface)) {
        QL_REQUIRE(!diffusionSpot.empty(), "diffusionSpot must not be empty");
        diffusionSpot_ = std::move(diffusionSpot);
        registerWith(diffusionSpot_);
    }

    Real BSplineLocalVolSurface::diffusionForward(Time t, Real surfaceForward) const {
        return diffusionSpot_.empty() ? surfaceForward :
                                        blackSurface_->forward(t, diffusionSpot_->value());
    }

    const Date& BSplineLocalVolSurface::referenceDate() const {
        return blackSurface_->referenceDate();
    }

    DayCounter BSplineLocalVolSurface::dayCounter() const { return blackSurface_->dayCounter(); }

    Date BSplineLocalVolSurface::maxDate() const { return blackSurface_->maxDate(); }

    Real BSplineLocalVolSurface::minStrike() const { return blackSurface_->minStrike(); }

    Real BSplineLocalVolSurface::maxStrike() const { return blackSurface_->maxStrike(); }

    Volatility BSplineLocalVolSurface::localVolImpl(Time t, Real underlyingLevel) const {
        if (t < 1e-14)
            t = 1e-14;
        const Real d =
            blackSurface_->pureDividendCoordinates() ? blackSurface_->dividendPV(t) : 0.0;
        if (d != 0.0 && underlyingLevel <= d)
            return 0.0;
        Real sigma;
        if (diffusionSpot_.empty()) {
            const Real x = blackSurface_->coordinate(t, underlyingLevel);
            sigma = std::sqrt(blackSurface_->localVariance(x, t));
        } else {
            const Real fwd = blackSurface_->forward(t);
            const Real fwdD = diffusionForward(t, fwd);
            QL_REQUIRE(fwd > d && fwdD > d, "forwards " << fwd << " (surface) and " << fwdD
                                                        << " (diffusion) at t=" << t
                                                        << " must be above D(t)=" << d);
            const Real y = std::log((underlyingLevel - d) / (fwdD - d));
            const Real shift = std::log((fwdD - d) / (fwd - d));
            sigma = std::sqrt(blackSurface_->localVariance(y, t, shift));
        }
        return d == 0.0 ? sigma : sigma * (underlyingLevel - d) / underlyingLevel;
    }

    Size BSplineLocalVolSurface::localVolSlice(Time t,
                                               const Array& underlyingLevels,
                                               Array& out) const {
        const Size n = underlyingLevels.size();
        QL_REQUIRE(out.size() == n,
                   "localVolSlice: output size " << out.size() << " != input size " << n);
        checkRange(t, true);
        if (t < 1e-14)
            t = 1e-14;
        // Forward and D(t) once per time, as localVolImpl() takes them.
        const Real fwd = blackSurface_->forward(t);
        const Real fwdD = diffusionForward(t, fwd);
        const Real d =
            blackSurface_->pureDividendCoordinates() ? blackSurface_->dividendPV(t) : 0.0;
        QL_REQUIRE(fwd > d, "localVolSlice: forward " << fwd << " not above D(t)=" << d);
        QL_REQUIRE(fwdD > d,
                   "localVolSlice: diffusion forward " << fwdD << " not above D(t)=" << d);
        // Sticky strike: the diffusion's coordinate and the surface's differ
        // by a constant (zero when anchored).
        const Real shift = diffusionSpot_.empty() ? 0.0 : std::log((fwdD - d) / (fwd - d));
        for (Size j = 0; j < n; ++j) {
            const Real s = underlyingLevels[j];
            // Below D(t) the pure spot cannot go: any finite x, zeroed below.
            out[j] = s > d ? std::log((s - d) / (fwdD - d)) : 0.0;
        }
        Size nIllegal =
            blackSurface_->localVarianceSlice(t, out.begin(), n, out.begin(), shift);
        for (Size j = 0; j < n; ++j) {
            const Real s = underlyingLevels[j];
            if (d != 0.0 && s <= d) {
                if (out[j] == Null<Real>())
                    --nIllegal;
                out[j] = 0.0;
            } else if (out[j] != Null<Real>()) {
                out[j] = std::sqrt(out[j]);
                if (d != 0.0)
                    out[j] *= (s - d) / s;
            }
        }
        return nIllegal;
    }

}
