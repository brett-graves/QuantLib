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

/*! \file fdmblackscholesstripsolver.hpp
    \brief One local-vol FD rollback for every vanilla of one expiry.

    In the Black-Scholes local-vol PDE the operator does not depend on
    the strike; only the terminal payoff and the exercise value do.  The
    strip solver therefore rolls back every option of one expiry, puts
    and calls alike, on one shared ln S mesh: one local-vol slice, one
    tridiagonal and one Thomas factorisation per time step, every option
    a right-hand-side column.  Between stopping times the PDE is linear,
    and the dividend jump and the American constraint act pointwise per
    column, so each column is exactly what FdBlackScholesVanillaEngine
    would compute for that option on the same mesh and time grid.

    It reproduces FdBlackScholesVanillaEngine's Douglas scheme, its
    stopping times (ex-dates, ex-dates + 1e-5, extra stopping times and
    Fdm1DimSolver's theta snapshot), the spot cash-dividend jump and the
    spline read-out of value and theta, so its values are the engine's
    on the same mesh and time grid.  Delta and gamma differ, because the
    spline's derivatives are wrong next to an American exercise boundary:
    they are the derivatives at the spot node of the polynomial in S
    through the nodes within two of it that lie on its side of the
    exercise boundary (a quartic away from it, one-sided next to it,
    exact inside the exercise region).  Spot must be a mesh node at least
    two nodes from either end; stripMesher() makes it one.

    Supported: local vol, cash or fractional dividends under the spot
    model, American or European exercise, no damping steps.  The exact
    cell average needs plain-vanilla payoffs.
*/

#ifndef quantlib_fdm_black_scholes_strip_solver_hpp
#define quantlib_fdm_black_scholes_strip_solver_hpp

#include <ql/instruments/dividendschedule.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/methods/finitedifferences/meshers/fdm1dmesher.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/time/date.hpp>
#include <vector>

namespace QuantLib {

    struct FdmBlackScholesStripResults {
        //! one entry per payoff, in input order
        std::vector<Real> value, delta, gamma, theta;
        //! local-vol points replaced by the illegal-local-vol overwrite
        Size illegalLocalVolCount = 0;
    };

    class FdmBlackScholesStripSolver {
      public:
        /*! \param mesher  ln S mesher shared by every payoff; see
                           stripMesher().
            \param tGrid   Douglas steps from maturity to today, before
                           stopping-time splits.
            \param illegalLocalVolOverwrite  local vol used where the
                           surface's local variance is illegal; negative
                           means raise instead (as the vanilla engine).
            \param stoppingTimes  extra stopping times; those outside
                           (0, maturity) are ignored.
            \param exactCellAverage  terminal values as the exact average
                           of each plain-vanilla payoff over its ln S cell.
                           False uses the vanilla engine's own Simpson
                           average (tolerance 5e-5 relative, at most 8
                           refinements, about 0.5 us per node and option),
                           which reproduces FdBlackScholesVanillaEngine.
        */
        FdmBlackScholesStripSolver(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process,
            const Date& maturityDate,
            std::vector<ext::shared_ptr<StrikedTypePayoff> > payoffs,
            bool american,
            DividendSchedule dividends,
            ext::shared_ptr<Fdm1dMesher> mesher,
            Size tGrid,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            std::vector<Time> stoppingTimes = {},
            bool exactCellAverage = false);

        FdmBlackScholesStripResults solve() const;

        /*! The strip mesh: the union over \p strikes of
            FdmBlackScholesMesher::xRange, concentrated at spot with
            density \p spotDensity relative to the range width, spot a
            required node.
        */
        static ext::shared_ptr<Fdm1dMesher> stripMesher(
            const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
            Time maturity,
            const std::vector<Real>& strikes,
            const DividendSchedule& dividends,
            Size xGrid,
            Real scaleFactor = 2.0,
            Real eps = 0.0001,
            Real spotDensity = 0.1);

      private:
        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        Date maturityDate_;
        std::vector<ext::shared_ptr<StrikedTypePayoff> > payoffs_;
        bool american_;
        DividendSchedule dividends_;
        ext::shared_ptr<Fdm1dMesher> mesher_;
        Size tGrid_;
        Real illegalLocalVolOverwrite_;
        std::vector<Time> stoppingTimes_;
        bool exactCellAverage_;
    };

}

#endif
