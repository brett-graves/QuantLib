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

#include <ql/termstructures/volatility/equityfx/puredividend.hpp>
#include <ql/errors.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace QuantLib {

    namespace {

        // Carry factor P(u) = Dr(u)/Dq(u): a dividend paid at u is worth
        // D P(u) / P(t) at t.
        Real carryFactor(const YieldTermStructure& r,
                         const YieldTermStructure& q, Time u) {
            return r.discount(u, true) / q.discount(u, true);
        }

        // Floor for the shifted strike: a strike at or below the PV of the
        // remaining dividends has no optionality left in the pure model.
        const Real minShiftedStrike = 1e-12;

    }

    Real cashDividendPV(const DividendSchedule& dividends,
                        const YieldTermStructure& riskFreeRate,
                        const YieldTermStructure& dividendYield,
                        const Date& referenceDate,
                        const DayCounter& dayCounter,
                        Time t) {
        if (dividends.empty())
            return 0.0;
        const Real pt = carryFactor(riskFreeRate, dividendYield, t);
        Real pv = 0.0;
        for (const auto& div : dividends) {
            const Date exDate = div->date();
            if (exDate <= referenceDate)
                continue;
            const Time ti = dayCounter.yearFraction(referenceDate, exDate);
            if (ti >= t)
                pv += div->amount() * carryFactor(riskFreeRate, dividendYield, ti) / pt;
        }
        return pv;
    }

    Real cashDividendForward(Real spot,
                             const DividendSchedule& dividends,
                             const YieldTermStructure& riskFreeRate,
                             const YieldTermStructure& dividendYield,
                             const Date& referenceDate,
                             const DayCounter& dayCounter,
                             Time t) {
        Real paid = 0.0;
        for (const auto& div : dividends) {
            const Date exDate = div->date();
            if (exDate <= referenceDate)
                continue;
            const Time ti = dayCounter.yearFraction(referenceDate, exDate);
            if (ti < t)
                paid += div->amount() * carryFactor(riskFreeRate, dividendYield, ti);
        }
        return (spot - paid) / carryFactor(riskFreeRate, dividendYield, t);
    }

    Volatility pureDividendBlackVol(Time t, Real strike, Real forward,
                                    Real dividendPV, Real wX) {
        QL_REQUIRE(t > 0.0, "pureDividendBlackVol: time must be > 0");
        QL_REQUIRE(wX >= 0.0, "pureDividendBlackVol: negative total variance " << wX);
        if (dividendPV == 0.0)
            return std::sqrt(wX / t);
        QL_REQUIRE(forward > dividendPV,
                   "pureDividendBlackVol: forward " << forward
                   << " not above the remaining dividends' PV " << dividendPV);
        const Real scale = forward - dividendPV;
        const Real kx = std::max((strike - dividendPV) / scale, minShiftedStrike);
        const Option::Type type = strike >= forward ? Option::Call : Option::Put;
        const Real price = scale * blackFormula(type, kx, 1.0, std::sqrt(wX), 1.0);
        if (!(price > 0.0))
            return 0.0;
        // Near the money the S vol is the X vol scaled by (F - D)/F.
        const Real guess = std::sqrt(wX) * scale / forward;
        const Real stdDev = blackFormulaImpliedStdDev(
            type, strike, forward, price, 1.0, 0.0, guess, 1e-12 * forward, 200);
        return stdDev / std::sqrt(t);
    }

    Real pureDividendBlackVolSensitivity(Time t, Real strike, Real forward,
                                         Real dividendPV, Real wX,
                                         Volatility blackVol) {
        if (dividendPV == 0.0)
            return 1.0;
        // Both legs are the same price: (F - D) Black(1, K_X, sigma_X sqrt t)
        // and Black(F, K, sigma_S sqrt t).  Their vegas give the ratio.
        const Real scale = forward - dividendPV;
        const Real kx = std::max((strike - dividendPV) / scale, minShiftedStrike);
        const Real sqrtT = std::sqrt(t);
        const NormalDistribution phi;
        const Real sdX = std::sqrt(wX);
        const Real sdS = blackVol * sqrtT;
        QL_REQUIRE(sdX > 0.0 && sdS > 0.0,
                   "pureDividendBlackVolSensitivity: zero vol at strike " << strike);
        const Real d1X = (-std::log(kx) + 0.5 * wX) / sdX;
        const Real d1S = (std::log(forward / strike) + 0.5 * sdS * sdS) / sdS;
        const Real vegaX = scale * phi(d1X) * sqrtT;
        const Real vegaS = forward * phi(d1S) * sqrtT;
        QL_REQUIRE(vegaS > 0.0,
                   "pureDividendBlackVolSensitivity: zero Black vega at strike " << strike);
        return vegaX / vegaS;
    }

    PureDividendFlatLocalVol::PureDividendFlatLocalVol(const Date& referenceDate,
                                                       Handle<YieldTermStructure> riskFreeRate,
                                                       Handle<YieldTermStructure> dividendYield,
                                                       DividendSchedule dividends,
                                                       Handle<Quote> sigmaX,
                                                       const DayCounter& dayCounter)
    : LocalVolTermStructure(referenceDate, Calendar(), Following, dayCounter),
      riskFreeRate_(std::move(riskFreeRate)), dividendYield_(std::move(dividendYield)),
      dividends_(std::move(dividends)), sigmaX_(std::move(sigmaX)) {
        QL_REQUIRE(!riskFreeRate_.empty(), "risk-free rate handle must not be empty");
        QL_REQUIRE(!dividendYield_.empty(), "dividend yield handle must not be empty");
        QL_REQUIRE(!sigmaX_.empty(), "sigmaX handle must not be empty");
        registerWith(riskFreeRate_);
        registerWith(dividendYield_);
        registerWith(sigmaX_);
    }

    Real PureDividendFlatLocalVol::dividendPV(Time t) const {
        return cashDividendPV(dividends_, *riskFreeRate_, *dividendYield_, referenceDate(),
                              dayCounter(), t);
    }

    Volatility PureDividendFlatLocalVol::localVolImpl(Time t, Real underlyingLevel) const {
        const Real d = dividendPV(t);
        if (underlyingLevel <= d)
            return 0.0;
        return sigmaX_->value() * (underlyingLevel - d) / underlyingLevel;
    }

    Size PureDividendFlatLocalVol::localVolSlice(Time t, const Array& underlyingLevels,
                                                 Array& out) const {
        const Size n = underlyingLevels.size();
        QL_REQUIRE(out.size() == n,
                   "localVolSlice: output size " << out.size() << " != input size " << n);
        checkRange(t, true);
        const Real d = dividendPV(t);
        const Real sigma = sigmaX_->value();
        for (Size j = 0; j < n; ++j) {
            const Real s = underlyingLevels[j];
            out[j] = s <= d ? 0.0 : sigma * (s - d) / s;
        }
        return 0;
    }

}
