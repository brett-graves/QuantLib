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

#include <ql/termstructures/volatility/equityfx/parametriclocalvolsurface.hpp>
#include <ql/errors.hpp>
#include <ql/utilities/null.hpp>
#include <cmath>
#include <utility>

namespace QuantLib {

    ParametricLocalVolSurface::ParametricLocalVolSurface(
            ext::shared_ptr<ParametricVolTermStructure> blackSurface,
            Handle<YieldTermStructure> riskFreeRate,
            Handle<YieldTermStructure> dividendYield,
            Handle<Quote> spot)
        : LocalVolTermStructure(blackSurface->businessDayConvention(),
                                blackSurface->dayCounter()),
          blackSurface_(std::move(blackSurface)),
          riskFreeRate_(std::move(riskFreeRate)),
          dividendYield_(std::move(dividendYield)),
          spot_(std::move(spot))
    {
        QL_REQUIRE(blackSurface_, "blackSurface must not be null");
        QL_REQUIRE(!riskFreeRate_.empty(),
                   "risk-free rate handle must not be empty");
        QL_REQUIRE(!dividendYield_.empty(),
                   "dividend yield handle must not be empty");
        QL_REQUIRE(!spot_.empty(), "spot handle must not be empty");
        registerWith(blackSurface_);
        registerWith(riskFreeRate_);
        registerWith(dividendYield_);
        registerWith(spot_);
    }

    const Date& ParametricLocalVolSurface::referenceDate() const {
        return blackSurface_->referenceDate();
    }

    DayCounter ParametricLocalVolSurface::dayCounter() const {
        return blackSurface_->dayCounter();
    }

    Date ParametricLocalVolSurface::maxDate() const {
        return blackSurface_->maxDate();
    }

    Real ParametricLocalVolSurface::minStrike() const {
        return blackSurface_->minStrike();
    }

    Real ParametricLocalVolSurface::maxStrike() const {
        return blackSurface_->maxStrike();
    }

    void ParametricLocalVolSurface::accept(AcyclicVisitor& v) {
        auto* v1 = dynamic_cast<Visitor<ParametricLocalVolSurface>*>(&v);
        if (v1 != nullptr)
            v1->visit(*this);
        else
            LocalVolTermStructure::accept(v);
    }

    Volatility ParametricLocalVolSurface::localVolImpl(
            Time t, Real underlyingLevel) const {
        if (t < 1e-14) t = 1e-14;
        // The coordinate comes from the Black surface itself, so its cash
        // dividends (forward, and D(t) in pure mode) are the ones the slices
        // were built against; a forward rebuilt here from S Dq/Dr drops them.
        const Real d =
            blackSurface_->pureDividendCoordinates() ? blackSurface_->dividendPV(t) : 0.0;
        if (d != 0.0 && underlyingLevel <= d)
            return 0.0;
        const Real x = blackSurface_->coordinate(t, underlyingLevel);
        const Real sigma = blackSurface_->localVol(x, t);
        // Unscaled without dividends to come, so that path stays bit-identical.
        return d == 0.0 ? sigma : sigma * (underlyingLevel - d) / underlyingLevel;
    }

    Size ParametricLocalVolSurface::localVolSlice(
            Time t, const Array& underlyingLevels, Array& out) const {
        const Size n = underlyingLevels.size();
        QL_REQUIRE(out.size() == n,
                   "localVolSlice: output size " << out.size()
                   << " != input size " << n);
        checkRange(t, true);
        // localVolImpl() for every point, with the forward taken once.
        if (t < 1e-14) t = 1e-14;
        // Forward and D(t) once per time, as localVolImpl() takes them.
        const Real fwd = blackSurface_->forward(t);
        const Real d =
            blackSurface_->pureDividendCoordinates() ? blackSurface_->dividendPV(t) : 0.0;
        QL_REQUIRE(fwd > d, "localVolSlice: forward " << fwd << " not above D(t)=" << d);
        for (Size j = 0; j < n; ++j) {
            const Real s = underlyingLevels[j];
            // Below D(t) the pure spot cannot go: any finite x, zeroed below.
            out[j] = s > d ? std::log((s - d) / (fwd - d)) : 0.0;
        }
        Size nIllegal =
            blackSurface_->localVarianceSlice(t, out.begin(), n, out.begin());
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

    std::vector<Volatility> ParametricLocalVolSurface::localVolGrid(
            const std::vector<Time>& times,
            const std::vector<Real>& underlyingLevels) const {
        // Row by row through localVolSlice(): the forward is taken once per
        // time and the Black surface's vectorised localVarianceSlice() does
        // the rest, ~2.5x cheaper per node than per-point localVol(k, t).
        // An illegal node raises, as the per-point path does.
        const Size nT = times.size();
        const Size nS = underlyingLevels.size();
        std::vector<Volatility> out(nT * nS);
        Array levels(underlyingLevels.begin(), underlyingLevels.end());
        Array row(nS);
        for (Size i = 0; i < nT; ++i) {
            localVolSlice(times[i], levels, row);
            Volatility* dst = out.data() + i * nS;
            for (Size j = 0; j < nS; ++j) {
                QL_REQUIRE(row[j] != Null<Real>(),
                           "ParametricLocalVolSurface::localVolGrid: illegal "
                           "local variance at t=" << times[i]
                           << ", S=" << underlyingLevels[j]);
                dst[j] = row[j];
            }
        }
        return out;
    }

} // namespace QuantLib
