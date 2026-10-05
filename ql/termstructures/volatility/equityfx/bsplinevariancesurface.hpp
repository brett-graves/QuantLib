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

/*! \file bsplinevariancesurface.hpp
    \brief Black vol surface from per-expiry B-splines in total variance
*/

#ifndef quantlib_bspline_variance_surface_hpp
#define quantlib_bspline_variance_surface_hpp

#include <ql/handle.hpp>
#include <ql/instruments/dividendschedule.hpp>
#include <ql/quote.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <array>

namespace QuantLib {

    //! Black vol surface built from per-expiry B-splines in total variance
    /*! Each pillar \f$ T_i \f$ carries a clamped knot vector and B-spline
        coefficients of degree \f$ p \f$ for the total variance
        \f$ w_i(k) = \sigma^2(T_i, k)\,T_i \f$ in log-forward moneyness
        \f$ k = \ln(K/F(T_i)) \f$.  A spline fitted elsewhere (chloride's
        Meniscus calibrator) is therefore reproduced exactly from its knots
        and coefficients.

        Wings: outside the knot range \f$ [k_L, k_R] \f$ the total variance
        continues linearly with the spline's end slope, so the wing is the
        same \f$ C^1 \f$ curve (and \f$ C^{p-1} \f$ when the fit imposes
        \f$ w'' = 0 \f$ at the ends).  Slopes steeper than Lee's moment bound
        \f$ |w'| \le 2 \f$ are rejected, not clamped; a wing that would turn
        the variance negative raises when evaluated there.

        Time: the forward is \f$ F(t) = (S - PV_{divs}(t))\,D_q(t)/D_r(t) \f$
        and \f$ k = \ln(K/F(t)) \f$.  Total variance is linear in \f$ t \f$ at
        fixed \f$ k \f$ between pillars, scales to zero at \f$ t = 0 \f$
        before the first pillar, and continues the last interval's slope
        after the last pillar.  Pillar coefficients that satisfy
        \f$ w_{i+1}(k) \ge w_i(k) \f$ for all \f$ k \f$ are therefore free of
        calendar arbitrage everywhere in time.

        localVariance() is Gatheral's form of Dupire's equation in
        log-forward moneyness, \f$ \sigma^2_{LV} = \partial_t w / g \f$ with
        \f$ g = (1 - k w'/2w)^2 - (w'^2/4)(1/w + 1/4) + w''/2 \f$, evaluated
        analytically.  It assumes a continuous forward (no discrete dividend
        jump inside the interval).

        \ingroup termstructures
    */
    class BSplineVarianceSurface : public BlackVolatilityTermStructure {
      public:
        //! Continuous dividend yield only
        BSplineVarianceSurface(const Date& referenceDate,
                               const std::vector<Date>& dates,
                               const std::vector<std::vector<Real>>& knots,
                               const std::vector<std::vector<Real>>& coefficients,
                               Natural degree,
                               Handle<Quote> spot,
                               Handle<YieldTermStructure> riskFreeRate,
                               Handle<YieldTermStructure> dividendYield,
                               const DayCounter& dc = Actual365Fixed());

        //! With discrete dividends
        BSplineVarianceSurface(const Date& referenceDate,
                               const std::vector<Date>& dates,
                               const std::vector<std::vector<Real>>& knots,
                               const std::vector<std::vector<Real>>& coefficients,
                               Natural degree,
                               Handle<Quote> spot,
                               Handle<YieldTermStructure> riskFreeRate,
                               Handle<YieldTermStructure> dividendYield,
                               DividendSchedule dividends,
                               const DayCounter& dc = Actual365Fixed());

        //! \name TermStructure interface
        //@{
        Date maxDate() const override;
        //@}
        //! \name VolatilityTermStructure interface
        //@{
        Real minStrike() const override;
        Real maxStrike() const override;
        //@}

        //! \name Inspectors
        //@{
        //! Forward used to map strike to log-moneyness at time t.
        Real forward(Time t) const;
        //! Pillar times, ascending.
        const std::vector<Time>& times() const { return times_; }
        Natural degree() const { return degree_; }
        //! Total variance of pillar i at log-moneyness k (wings included).
        Real sliceTotalVariance(Size i, Real k) const;
        //! (w, dw/dk, d2w/dk2) of pillar i at k (wings included).
        std::vector<Real> sliceTotalVarianceDerivatives(Size i, Real k) const;
        //! Total variance at (k, t) under the time rule above.
        Real totalVariance(Real k, Time t) const;
        //! Local variance at (k, t); raises where g <= 0 or w <= 0.
        Real localVariance(Real k, Time t) const;
        //@}

        //! \name Visitability
        //@{
        void accept(AcyclicVisitor&) override;
        //@}

      protected:
        Volatility blackVolImpl(Time t, Real strike) const override;

      private:
        struct Slice {
            std::vector<Real> knots;
            std::vector<Real> coefficients;
            Real kL, kR;
            std::array<Real, 3> left, right;  // (w, w', w'') at the knot-range ends
        };
        void initialize(const std::vector<Date>& dates,
                        const std::vector<std::vector<Real>>& knots,
                        const std::vector<std::vector<Real>>& coefficients);
        std::array<Real, 3> inside(const Slice& s, Real k) const;
        std::array<Real, 3> derivatives(Size i, Real k) const;
        //! Bracketing pillars and weight for time t (lo == hi before the first pillar).
        void bracket(Time t, Size& lo, Size& hi, Real& a) const;

        Natural degree_;
        std::vector<Time> times_;
        std::vector<Slice> slices_;
        Handle<Quote> spot_;
        Handle<YieldTermStructure> riskFreeRate_;
        Handle<YieldTermStructure> dividendYield_;
        DividendSchedule dividends_;
    };

}

#endif
