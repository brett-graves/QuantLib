/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008 Ralph Schreyer
 Copyright (C) 2008, 2009 Klaus Spanderen

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

/*! \file fdmdividendhandler.hpp
    \brief dividend handler for fdm method for one equity direction
*/

#ifndef quantlib_fdm_dividend_handler_hpp
#define quantlib_fdm_dividend_handler_hpp

#include <ql/instruments/dividendschedule.hpp>
#include <ql/math/matrix.hpp>
#include <ql/methods/finitedifferences/stepcondition.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmesher.hpp>
#include <array>
#include <vector>

namespace QuantLib {
    
    class DayCounter;

    namespace detail {

        /*! The cash-dividend jump V(s) <- V(max(s_0, s - D(s))) on a fixed
            grid s_0 < ... < s_{n-1} (spot dividend model).

            The value is read between nodes with a 4-point Lagrange cubic
            in ln s, clamped to the two bracketing node values.  Linear
            interpolation of a convex value function reads high by
            O(h^2) at every ex-date, and the error accumulates over every
            dividend in the option's life (chloride #593: +$0.15 on a 2y
            SPY call at 150 nodes); the cubic removes that term.  The
            clamp keeps it from overshooting near a payoff kink when the
            ex-date is close to expiry; it binds only where the data is
            not locally monotone and convex.

            The stencil and weights depend on the grid and the dividend
            only, so they are built once and every application costs four
            multiply-adds per node and column.
        */
        class FdmDividendJump {
          public:
            FdmDividendJump(const std::vector<Real>& s, const Dividend& div);
            //! out[k] = jumped in[k]; in and out must not alias
            void apply(const Array& in, Array& out) const;
            //! the same for every column of a nodes x columns matrix
            void apply(const Matrix& in, Matrix& out) const;

          private:
            std::vector<Size> first_, lo_;
            std::vector<std::array<Real, 4> > w_;
        };

    }

    class FdmDividendHandler : public StepCondition<Array> {
      public:
        FdmDividendHandler(const DividendSchedule& schedule,
                           const ext::shared_ptr<FdmMesher>& mesher,
                           const Date& referenceDate,
                           const DayCounter& dayCounter,
                           Size equityDirection);

        void applyTo(Array& a, Time t) const override;

        const std::vector<Time>& dividendTimes() const;
        const std::vector<Date>& dividendDates() const;
        // Pre-computed cash-equivalent amounts. For FixedDividend this is
        // the dollar amount; for FractionalDividend without a preset
        // nominal it's 0.0 (the true per-node drop is computed at runtime
        // in applyTo via Dividend::amount(underlying)).
        const std::vector<Real>& dividends() const;

      private:
        Array x_; // grid-equity values in physical units

        std::vector<Time> dividendTimes_;
        std::vector<Date> dividendDates_;
        std::vector<Real> dividends_;
        // Original Dividend pointers, kept so applyTo can dispatch to the
        // spot-aware ``amount(underlying)`` overload for FractionalDividend.
        // For FixedDividend the overload returns the same constant amount;
        // for FractionalDividend it returns ``rate * underlying``, applied
        // per-grid-node at the ex-date so the spot drop scales with the
        // simulated spot, not the spot-today nominal.
        std::vector<ext::shared_ptr<Dividend>> dividendCashflows_;
        const ext::shared_ptr<FdmMesher> mesher_;
        const Size equityDirection_;
        // one precomputed jump per dividend, parallel to dividendTimes_
        std::vector<detail::FdmDividendJump> jumps_;
    };
}
#endif
