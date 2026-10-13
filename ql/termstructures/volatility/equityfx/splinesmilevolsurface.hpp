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

/*! \file splinesmilevolsurface.hpp
    \brief Black vol surface from per-expiry natural cubic smiles in log-moneyness
*/

#ifndef quantlib_spline_smile_vol_surface_hpp
#define quantlib_spline_smile_vol_surface_hpp

#include <ql/handle.hpp>
#include <ql/instruments/dividendschedule.hpp>
#include <ql/math/interpolation.hpp>
#include <ql/quote.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

namespace QuantLib {

    //! Black vol surface built from per-expiry natural cubic smiles
    /*! Each pillar \f$ T_i \f$ carries nodes \f$ (k_{ij}, \sigma_{ij}) \f$ in
        log-forward moneyness \f$ k = \ln(K/F(T_i)) \f$.  Inside the node range
        the smile is the natural cubic spline through the nodes
        (CubicInterpolation::Spline with zero second derivative at both
        ends), so a natural cubic fitted elsewhere on the same nodes — e.g.
        chloride's band spline — is reproduced exactly from its benchmarks.

        Wings: beyond the outer nodes the total variance
        \f$ w = \sigma^2 T \f$ continues linearly in \f$ k \f$ with the
        spline's end slope \f$ \partial w/\partial k \f$, clamped to
        \f$ [-s_{max}, 0] \f$ on the left and \f$ [0, s_{max}] \f$ on the
        right (Lee's moment bound is \f$ s_{max} = 2 \f$).  Wings therefore
        never decrease away from the nodes; the join is \f$ C^1 \f$ unless the
        clamp binds.

        Time: the forward is \f$ F(t) = (S - PV_{divs}(t))\,D_q(t)/D_r(t) \f$
        and \f$ k = \ln(K/F(t)) \f$.  Total variance is linear in \f$ t \f$ at
        fixed \f$ k \f$ between pillars, scales to zero at \f$ t = 0 \f$ before
        the first pillar, and continues the last interval's slope after the
        last pillar — the conventions of ParametricVolTermStructure.

        \ingroup termstructures
    */
    class SplineSmileVolSurface : public BlackVolatilityTermStructure {
      public:
        //! Continuous dividend yield only
        SplineSmileVolSurface(const Date& referenceDate,
                              const std::vector<Date>& dates,
                              const std::vector<std::vector<Real>>& kNodes,
                              const std::vector<std::vector<Real>>& volNodes,
                              Handle<Quote> spot,
                              Handle<YieldTermStructure> riskFreeRate,
                              Handle<YieldTermStructure> dividendYield,
                              const DayCounter& dc = Actual365Fixed(),
                              Real maxWingSlope = 2.0);

        //! With discrete dividends
        SplineSmileVolSurface(const Date& referenceDate,
                              const std::vector<Date>& dates,
                              const std::vector<std::vector<Real>>& kNodes,
                              const std::vector<std::vector<Real>>& volNodes,
                              Handle<Quote> spot,
                              Handle<YieldTermStructure> riskFreeRate,
                              Handle<YieldTermStructure> dividendYield,
                              DividendSchedule dividends,
                              const DayCounter& dc = Actual365Fixed(),
                              Real maxWingSlope = 2.0);

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
        //! Total variance of pillar i at log-moneyness k (wings included).
        Real sliceTotalVariance(Size i, Real k) const;
        //! Total variance at (k, t) under the time rule above.
        Real totalVariance(Real k, Time t) const;
        //@}

        //! \name Visitability
        //@{
        void accept(AcyclicVisitor&) override;
        //@}

      protected:
        Volatility blackVolImpl(Time t, Real strike) const override;

      private:
        struct Slice {
            std::vector<Real> k;
            std::vector<Real> vol;
            Interpolation smile;  // holds iterators into k / vol: never copy a built Slice
            Real wLeft, wRight;   // total variance at the outer nodes
            Real sLeft, sRight;   // clamped wing slopes dw/dk
        };
        void initialize(const std::vector<Date>& dates,
                        const std::vector<std::vector<Real>>& kNodes,
                        const std::vector<std::vector<Real>>& volNodes);

        std::vector<Time> times_;
        std::vector<Slice> slices_;
        Handle<Quote> spot_;
        Handle<YieldTermStructure> riskFreeRate_;
        Handle<YieldTermStructure> dividendYield_;
        DividendSchedule dividends_;
        Real maxWingSlope_;
    };

}

#endif
