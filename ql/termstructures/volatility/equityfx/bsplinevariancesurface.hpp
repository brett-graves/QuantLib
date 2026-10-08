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
#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
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

        Time: the forward is that of a spot paying the cash dividends
        (cashDividendForward: each dividend grows at r - q from its ex-date,
        as in the FD engines' spot dividend model) and
        \f$ k = \ln(K/F(t)) \f$.  Total variance is linear in \f$ t \f$ at
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

        Pure-dividend coordinates (setPureDividendCoordinates(true)): the
        slices are smiles of the pure process X in
        \f$ x = \ln((K - D(t))/(F(t) - D(t))) \f$, with \f$ D(t) \f$ the
        dividends still to come (Buehler; puredividend.hpp).  blackVol()
        returns the Black vol on (F, K) of the same price, and the local vol
        (BSplineLocalVolSurface) is \f$ \sigma_X (S - D)/S \f$, consistent
        with cash dividend drops.  Without cash dividends both modes are
        identical.

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
        //! Forward at time t of the given spot paying this surface's dividends
        //! on this surface's curves.
        Real forward(Time t, Real spot) const;
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
        /*! With a nonzero \p shift, k is the log-moneyness of a diffusion
            whose (pure) forward differs from this surface's by a factor
            \f$ e^{s} \f$ constant between dividend dates, and the slices
            are read at \f$ k + s \f$: the smile held fixed in absolute
            strike (sticky strike).  Gatheral's \f$ g \f$ then takes the
            diffusion's own k, and \f$ \partial_t w \f$ is unchanged
            because \f$ s \f$ does not move with t. */
        Real localVariance(Real k, Time t, Real shift = 0.0) const;
        //! localVariance() at one time for n coordinates.
        /*! Where localVariance() would raise, out[i] is Null<Real>()
            instead; returns the number of such points.  k and out may be
            the same array. */
        Size localVarianceSlice(Time t, const Real* k, Size n, Real* out,
                                Real shift = 0.0) const;
        //! PV at t of the cash dividends still to come (zero without any).
        Real dividendPV(Time t) const;
        //! Strike coordinate the slices are read at: k, or x in pure mode.
        Real coordinate(Time t, Real strike) const;
        bool pureDividendCoordinates() const { return pureDividend_; }
        //@}

        //! \name Modifiers
        //@{
        void setPureDividendCoordinates(bool pure);
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
        bool pureDividend_ = false;
    };

    //! Analytic local vol of a BSplineVarianceSurface.
    /*! Gatheral's Dupire form on the surface's own slices, at the coordinate
        they were fitted in: the forward and, in pure-dividend mode, D(t)
        come from the surface, and the local vol of S is
        sigma_X(x, t) (S - D)/S, zero at or below D(t).  localVolSlice()
        computes the bracket and forward once per time.

        With a diffusion spot, the local vol is that of a diffusion started
        at that spot instead of the surface's own, with the smile held fixed
        in absolute strike (sticky strike).  Rates, carry and dividends stay
        the surface's.  The diffusion's (pure) forward
        \f$ F_d - D \f$ is then \f$ (S_d - D_0)/(S - D_0) \f$ times the
        surface's, so its coordinate y and the surface's x differ by the
        constant \f$ s = \ln((F_d - D)/(F - D)) \f$ between dividend dates
        and Dupire is exact in y with the slices read at y + s (see
        BSplineVarianceSurface::localVariance).  At the surface's own spot
        s = 0 and the result is the anchored local vol.
    */
    class BSplineLocalVolSurface : public LocalVolTermStructure {
      public:
        explicit BSplineLocalVolSurface(ext::shared_ptr<BSplineVarianceSurface> blackSurface);
        //! Local vol of a diffusion started at \p diffusionSpot (sticky strike).
        BSplineLocalVolSurface(ext::shared_ptr<BSplineVarianceSurface> blackSurface,
                               Handle<Quote> diffusionSpot);

        const Date& referenceDate() const override;
        DayCounter dayCounter() const override;
        Date maxDate() const override;
        Real minStrike() const override;
        Real maxStrike() const override;

        Size localVolSlice(Time t, const Array& underlyingLevels, Array& out) const override;

        const ext::shared_ptr<BSplineVarianceSurface>& blackSurface() const {
            return blackSurface_;
        }

      protected:
        Volatility localVolImpl(Time t, Real underlyingLevel) const override;

      private:
        //! Forward of the priced diffusion at t (the surface's own when anchored).
        Real diffusionForward(Time t, Real surfaceForward) const;

        ext::shared_ptr<BSplineVarianceSurface> blackSurface_;
        Handle<Quote> diffusionSpot_;
    };

}

#endif
