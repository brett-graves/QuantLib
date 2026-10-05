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
#include <ql/patterns/visitor.hpp>
#include <ql/termstructures/volatility/equityfx/splinesmilevolsurface.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace QuantLib {

    SplineSmileVolSurface::SplineSmileVolSurface(
        const Date& referenceDate,
        const std::vector<Date>& dates,
        const std::vector<std::vector<Real>>& kNodes,
        const std::vector<std::vector<Real>>& volNodes,
        Handle<Quote> spot,
        Handle<YieldTermStructure> riskFreeRate,
        Handle<YieldTermStructure> dividendYield,
        const DayCounter& dc,
        Real maxWingSlope)
    : SplineSmileVolSurface(referenceDate, dates, kNodes, volNodes, std::move(spot),
                            std::move(riskFreeRate), std::move(dividendYield),
                            DividendSchedule(), dc, maxWingSlope) {}

    SplineSmileVolSurface::SplineSmileVolSurface(
        const Date& referenceDate,
        const std::vector<Date>& dates,
        const std::vector<std::vector<Real>>& kNodes,
        const std::vector<std::vector<Real>>& volNodes,
        Handle<Quote> spot,
        Handle<YieldTermStructure> riskFreeRate,
        Handle<YieldTermStructure> dividendYield,
        DividendSchedule dividends,
        const DayCounter& dc,
        Real maxWingSlope)
    : BlackVolatilityTermStructure(referenceDate, Calendar(), Following, dc),
      spot_(std::move(spot)), riskFreeRate_(std::move(riskFreeRate)),
      dividendYield_(std::move(dividendYield)), dividends_(std::move(dividends)),
      maxWingSlope_(maxWingSlope) {
        QL_REQUIRE(!spot_.empty() && !riskFreeRate_.empty() && !dividendYield_.empty(),
                   "spot/rate/dividend handles must not be empty");
        QL_REQUIRE(maxWingSlope_ >= 0.0,
                   "maxWingSlope must be >= 0, got " << maxWingSlope_);
        registerWith(spot_);
        registerWith(riskFreeRate_);
        registerWith(dividendYield_);
        initialize(dates, kNodes, volNodes);
    }

    void SplineSmileVolSurface::initialize(
        const std::vector<Date>& dates,
        const std::vector<std::vector<Real>>& kNodes,
        const std::vector<std::vector<Real>>& volNodes) {
        const Size N = dates.size();
        QL_REQUIRE(N > 0, "at least one pillar is required");
        QL_REQUIRE(kNodes.size() == N && volNodes.size() == N,
                   "dates (" << N << "), kNodes (" << kNodes.size()
                   << ") and volNodes (" << volNodes.size() << ") sizes differ");

        std::vector<Size> order(N);
        std::iota(order.begin(), order.end(), Size(0));
        std::sort(order.begin(), order.end(),
                  [&dates](Size a, Size b) { return dates[a] < dates[b]; });

        times_.resize(N);
        // Size the slice vector once: each Slice's interpolation keeps
        // iterators into its own node vectors, so slices must not move.
        slices_ = std::vector<Slice>(N);
        for (Size i = 0; i < N; ++i) {
            const Size src = order[i];
            const Time T = dayCounter().yearFraction(referenceDate(), dates[src]);
            QL_REQUIRE(T > 0.0, "pillar " << dates[src] << " is not after the reference date");
            QL_REQUIRE(i == 0 || T > times_[i - 1],
                       "duplicate or non-increasing pillar at " << dates[src]);
            times_[i] = T;

            const std::vector<Real>& k = kNodes[src];
            const std::vector<Real>& v = volNodes[src];
            QL_REQUIRE(k.size() == v.size() && k.size() >= 2,
                       "pillar " << dates[src] << ": need >= 2 nodes with matching "
                       "k/vol sizes, got " << k.size() << "/" << v.size());
            for (Size j = 0; j < k.size(); ++j) {
                QL_REQUIRE(std::isfinite(k[j]) && std::isfinite(v[j]) && v[j] > 0.0,
                           "pillar " << dates[src] << " node " << j
                           << ": k and vol must be finite and vol > 0");
                QL_REQUIRE(j == 0 || k[j] > k[j - 1],
                           "pillar " << dates[src] << ": k nodes must be strictly increasing");
            }

            Slice& s = slices_[i];
            s.k = k;
            s.vol = v;
            s.smile = CubicNaturalSpline(s.k.begin(), s.k.end(), s.vol.begin());
            s.smile.update();

            const Real kL = s.k.front(), kR = s.k.back();
            const Real vL = s.vol.front(), vR = s.vol.back();
            s.wLeft = vL * vL * T;
            s.wRight = vR * vR * T;
            // dw/dk = 2 σ σ' T at the outer nodes, clamped so wings rise and
            // respect the moment bound.
            const Real dwL = 2.0 * vL * s.smile.derivative(kL) * T;
            const Real dwR = 2.0 * vR * s.smile.derivative(kR) * T;
            s.sLeft = std::min(0.0, std::max(dwL, -maxWingSlope_));
            s.sRight = std::max(0.0, std::min(dwR, maxWingSlope_));
        }
    }

    Date SplineSmileVolSurface::maxDate() const { return Date::maxDate(); }

    Real SplineSmileVolSurface::minStrike() const { return 0.0; }

    Real SplineSmileVolSurface::maxStrike() const { return QL_MAX_REAL; }

    Real SplineSmileVolSurface::forward(Time t) const {
        const Real S = spot_->value();
        const Real df = riskFreeRate_->discount(t);
        const Real dq = dividendYield_->discount(t);
        Real pvDivs = 0.0;
        if (!dividends_.empty()) {
            const Date refDate = referenceDate();
            const DayCounter dc = dayCounter();
            for (const auto& div : dividends_) {
                const Date exDate = div->date();
                if (exDate <= refDate)
                    continue;
                const Time tDiv = dc.yearFraction(refDate, exDate);
                if (tDiv < t)
                    pvDivs += div->amount() * riskFreeRate_->discount(tDiv);
            }
        }
        return (S - pvDivs) * dq / df;
    }

    Real SplineSmileVolSurface::sliceTotalVariance(Size i, Real k) const {
        QL_REQUIRE(i < slices_.size(), "pillar index " << i << " out of range");
        const Slice& s = slices_[i];
        if (k < s.k.front())
            return s.wLeft + s.sLeft * (k - s.k.front());
        if (k > s.k.back())
            return s.wRight + s.sRight * (k - s.k.back());
        const Real v = s.smile(k);
        return v * v * times_[i];
    }

    Real SplineSmileVolSurface::totalVariance(Real k, Time t) const {
        QL_REQUIRE(t > 0.0, "time must be > 0");
        const Size N = times_.size();
        if (t <= times_.front())
            return sliceTotalVariance(0, k) * t / times_.front();
        if (N == 1)
            return sliceTotalVariance(0, k) * t / times_.front();
        Size lo, hi;
        if (t >= times_.back()) {
            lo = N - 2;
            hi = N - 1;
        } else {
            hi = static_cast<Size>(std::upper_bound(times_.begin(), times_.end(), t)
                                   - times_.begin());
            lo = hi - 1;
        }
        const Real a = (t - times_[lo]) / (times_[hi] - times_[lo]);
        return (1.0 - a) * sliceTotalVariance(lo, k) + a * sliceTotalVariance(hi, k);
    }

    Volatility SplineSmileVolSurface::blackVolImpl(Time t, Real strike) const {
        if (t < 1e-14)
            t = 1e-14;
        const Real k = std::log(strike / forward(t));
        const Real w = totalVariance(k, t);
        QL_REQUIRE(w >= 0.0, "negative total variance " << w << " at (k=" << k
                             << ", t=" << t << ")");
        return std::sqrt(w / t);
    }

    void SplineSmileVolSurface::accept(AcyclicVisitor& v) {
        auto* v1 = dynamic_cast<Visitor<SplineSmileVolSurface>*>(&v);
        if (v1 != nullptr)
            v1->visit(*this);
        else
            BlackVolatilityTermStructure::accept(v);
    }

}
