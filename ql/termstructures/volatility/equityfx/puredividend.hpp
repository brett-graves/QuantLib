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

/*! \file puredividend.hpp
    \brief Cash-dividend forward and pure-dividend (Buehler) coordinates.

    One convention for every cash-dividend term structure in the fork, the
    same one QuantLib's spot-model FD engine diffuses: between ex-dates the
    spot drifts at r - q, and on an ex-date it drops by the dividend.  Hence

        F(t) = (S - sum_{t_i < t} D_i P(t_i)) / P(t),   P(u) = Dr(u)/Dq(u),

    and the PV at t of the dividends still to come,

        D(t) = sum_{t_i >= t} D_i P(t_i) / P(t),

    grows at r - q between ex-dates and drops by D_i at t_i, exactly as the
    spot does.  A dividend going ex at t itself is still to come at t (the
    spot is cum-dividend), matching the strict ``t_i < t`` of the forward.

    Pure-dividend coordinates (H. Buehler, "Volatility and Dividends",
    2010): S_t = (F(t) - D(t)) X_t + D(t) with X a driftless positive
    martingale, X_0 = 1.  Options on S are options on X at the shifted
    strike K_X = (K - D(t)) / (F(t) - D(t)); a smile parameterised in
    x = ln K_X and interpolated in time at fixed x is consistent with the
    cash-dividend drops, which a smile in ln(K/F) is not.  The local vol of
    S follows from the local vol of X:

        sigma_S(t, S) = sigma_X(t, x) (S - D(t)) / S.

    Only fixed-cash dividends belong here: proportional dividends are part
    of the dividend-yield curve.
*/

#ifndef quantlib_pure_dividend_hpp
#define quantlib_pure_dividend_hpp

#include <ql/handle.hpp>
#include <ql/instruments/dividendschedule.hpp>
#include <ql/quote.hpp>
#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/daycounter.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

namespace QuantLib {

    //! PV at t of the cash dividends going ex at or after t (see file docs).
    Real cashDividendPV(const DividendSchedule& dividends,
                        const YieldTermStructure& riskFreeRate,
                        const YieldTermStructure& dividendYield,
                        const Date& referenceDate,
                        const DayCounter& dayCounter,
                        Time t);

    //! Forward of a spot paying the given cash dividends (see file docs).
    Real cashDividendForward(Real spot,
                             const DividendSchedule& dividends,
                             const YieldTermStructure& riskFreeRate,
                             const YieldTermStructure& dividendYield,
                             const Date& referenceDate,
                             const DayCounter& dayCounter,
                             Time t);

    //! Black vol on the actual forward of the pure-dividend price.
    /*! Price of the out-of-the-money option on S with strike K at t when
        X has total variance ``wX`` at x = ln((K - D)/(F - D)), inverted to
        the Black vol on (F, K).  Raises when the inversion fails; there is
        no fallback vol. */
    Volatility pureDividendBlackVol(Time t, Real strike, Real forward,
                                    Real dividendPV, Real wX);

    //! d(sigma_Black)/d(sigma_X) at the same point, for chain-rule gradients.
    Real pureDividendBlackVolSensitivity(Time t, Real strike, Real forward,
                                         Real dividendPV, Real wX,
                                         Volatility blackVol);

    //! Local vol of S when the pure process X has a flat vol sigma_X.
    /*! sigma_S(t, S) = sigma_X (S - D(t)) / S, zero at or below D(t)
        (the pure-dividend spot never falls below the dividends still to
        come).  This is the flat-vol member of the pure-dividend family:
        inverting an American quote on it gives the X vol that a
        pure-dividend surface is fitted to, under the same dividend model
        the surface's local vol prices with. */
    class PureDividendFlatLocalVol : public LocalVolTermStructure {
      public:
        PureDividendFlatLocalVol(const Date& referenceDate,
                                 Handle<YieldTermStructure> riskFreeRate,
                                 Handle<YieldTermStructure> dividendYield,
                                 DividendSchedule dividends,
                                 Handle<Quote> sigmaX,
                                 const DayCounter& dayCounter = Actual365Fixed());
        Date maxDate() const override { return Date::maxDate(); }
        Real minStrike() const override { return 0.0; }
        Real maxStrike() const override { return QL_MAX_REAL; }
        //! PV at t of the dividends still to come.
        Real dividendPV(Time t) const;
        Size localVolSlice(Time t, const Array& underlyingLevels, Array& out) const override;

      protected:
        Volatility localVolImpl(Time t, Real underlyingLevel) const override;

      private:
        Handle<YieldTermStructure> riskFreeRate_, dividendYield_;
        DividendSchedule dividends_;
        Handle<Quote> sigmaX_;
    };

}

#endif
