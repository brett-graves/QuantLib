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
        // Forward at t — uses the local-vol surface's own r/q/S handles.
        // Caller is responsible for passing the same handles used by the
        // underlying parametric surface so the two forwards agree.
        DiscountFactor dr = riskFreeRate_->discount(t, true);
        DiscountFactor dq = dividendYield_->discount(t, true);
        Real fwd = spot_->value() * dq / dr;
        Real k = std::log(underlyingLevel / fwd);
        return blackSurface_->localVol(k, t);
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
        DiscountFactor dr = riskFreeRate_->discount(t, true);
        DiscountFactor dq = dividendYield_->discount(t, true);
        Real fwd = spot_->value() * dq / dr;
        for (Size j = 0; j < n; ++j)
            out[j] = std::log(underlyingLevels[j] / fwd);
        Size nIllegal =
            blackSurface_->localVarianceSlice(t, out.begin(), n, out.begin());
        for (Size j = 0; j < n; ++j)
            if (out[j] != Null<Real>())
                out[j] = std::sqrt(out[j]);
        return nIllegal;
    }

    std::vector<Volatility> ParametricLocalVolSurface::localVolGrid(
            const std::vector<Time>& times,
            const std::vector<Real>& underlyingLevels) const {
        const Size nT = times.size();
        const Size nS = underlyingLevels.size();
        std::vector<Volatility> out(nT * nS);
        const Real spotVal = spot_->value();
        std::vector<Real> logS(nS);
        for (Size j = 0; j < nS; ++j)
            logS[j] = std::log(underlyingLevels[j]);
        for (Size i = 0; i < nT; ++i) {
            Time t = times[i];
            if (t < 1e-14) t = 1e-14;
            DiscountFactor dr = riskFreeRate_->discount(t, true);
            DiscountFactor dq = dividendYield_->discount(t, true);
            Real logFwd = std::log(spotVal) + std::log(dq / dr);
            Volatility* row = out.data() + i * nS;
            for (Size j = 0; j < nS; ++j) {
                Real k = logS[j] - logFwd;
                row[j] = blackSurface_->localVol(k, t);
            }
        }
        return out;
    }

} // namespace QuantLib
