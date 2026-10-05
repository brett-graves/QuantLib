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

/*! \file parametriclocalvolsurface.hpp
    \brief Analytic Dupire local volatility from a Klassen parametric
           Black vol surface.

    Mirrors EssviLocalVolSurface: a thin LocalVolTermStructure that
    forwards to ParametricVolTermStructure::localVol(k, t).  Closed-form
    derivatives flow from the shape's analytic f, f', f'' (S3, JW) — the
    FD bumping done by the generic LocalVolSurface is avoided entirely.
    For Python-supplied shapes that don't override d2fdz2, the surface
    falls back to a single FD call on dfdz (one evaluation pair), still
    much cheaper than the multi-bump Dupire FD inside LocalVolSurface.

    Forward consistency: the local-vol surface owns its own (r, q, S)
    handles; the caller is responsible for passing the same handles that
    constructed the underlying ParametricVolTermStructure so the two
    forwards agree.
*/

#ifndef quantlib_parametric_local_vol_surface_hpp
#define quantlib_parametric_local_vol_surface_hpp

#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
#include <ql/termstructures/volatility/equityfx/parametricvoltermstructure.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/quote.hpp>
#include <ql/handle.hpp>

namespace QuantLib {

    //! Analytic Dupire local vol from a parametric Black vol surface.
    class ParametricLocalVolSurface : public LocalVolTermStructure {
      public:
        ParametricLocalVolSurface(
            ext::shared_ptr<ParametricVolTermStructure> blackSurface,
            Handle<YieldTermStructure> riskFreeRate,
            Handle<YieldTermStructure> dividendYield,
            Handle<Quote> spot);

        //! \name TermStructure interface
        //@{
        const Date& referenceDate() const override;
        DayCounter dayCounter() const override;
        Date maxDate() const override;
        //@}

        //! \name VolatilityTermStructure interface
        //@{
        Real minStrike() const override;
        Real maxStrike() const override;
        //@}

        //! \name Visitability
        //@{
        void accept(AcyclicVisitor&) override;
        //@}

        //! Underlying parametric Black surface (for inspection/tests).
        const ext::shared_ptr<ParametricVolTermStructure>& blackSurface() const {
            return blackSurface_;
        }

        /*! Batched evaluation on a tensor-product (t, S) grid.

            Returns ``out[i * underlyingLevels.size() + j]`` = analytic
            ``localVol(times[i], underlyingLevels[j])``, evaluated one
            time row at a time through localVolSlice().  Raises on an
            illegal (negative or undefined) local variance.  See
            EssviLocalVolSurface::localVolGrid for the use case rationale.
        */
        std::vector<Volatility> localVolGrid(
            const std::vector<Time>& times,
            const std::vector<Real>& underlyingLevels) const;

        /*! One FD time step's worth of local vols: the forward is
            computed once and the Black surface's localVarianceSlice()
            does the rest.  Same values as localVol(t, S, true). */
        Size localVolSlice(Time t,
                           const Array& underlyingLevels,
                           Array& out) const override;

      protected:
        Volatility localVolImpl(Time t, Real underlyingLevel) const override;

      private:
        ext::shared_ptr<ParametricVolTermStructure> blackSurface_;
        Handle<YieldTermStructure>                  riskFreeRate_;
        Handle<YieldTermStructure>                  dividendYield_;
        Handle<Quote>                               spot_;
    };

} // namespace QuantLib

#endif
