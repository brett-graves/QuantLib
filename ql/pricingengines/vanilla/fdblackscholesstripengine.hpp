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

/*! \file fdblackscholesstripengine.hpp
    \brief Local-vol FD engine that prices every option of an expiry in
           one strip solve and serves them from a cache.
*/

#ifndef quantlib_fd_black_scholes_strip_engine_hpp
#define quantlib_fd_black_scholes_strip_engine_hpp

#include <ql/exercise.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholesstripsolver.hpp>
#include <map>
#include <tuple>

namespace QuantLib {

    //! Strip local-vol FD engine for vanilla options
    /*! One instance is shared by the options of an underlier.  The first
        calculate() for an (expiry, exercise) group solves every option
        declared for that group, puts and calls together, with
        FdmBlackScholesStripSolver on one stripMesher() mesh; the rest of
        the group reads the cached results.  Any notification from the
        process invalidates every group.

        Declare a chain with declare() before pricing it.  An option that
        was not declared still prices: it joins its group, and the group
        is solved again.

        The time grid is max(tGrid, round(minStepsPerYear * T)) steps.
        Terminal values are exact cell averages of the payoffs, not the
        vanilla engine's Simpson approximation.

        The mesh is FdmBlackScholesStripSolver::stripMesher(): with
        \p coreStdDevs null, concentrated at spot with relative density
        \p spotConcentrationDensity; otherwise a uniform core of
        \p coreStdDevs ATM standard deviations holding \p coreFraction
        of the nodes, graded out to the range ends.
        Cash and fractional dividends use the spot model.  Pricing a
        single option this way costs a whole strip; use
        FdBlackScholesVanillaEngine for that.

        \ingroup vanillaengines
    */
    class FdBlackScholesStripEngine : public VanillaOption::engine {
      public:
        explicit FdBlackScholesStripEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process,
            DividendSchedule dividends = {},
            Size xGrid = 300,
            Size tGrid = 100,
            Real minStepsPerYear = 100.0,
            Real mesherScaleFactor = 2.0,
            Real mesherEps = 0.0001,
            Real spotConcentrationDensity = 0.1,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            std::vector<Time> stoppingTimes = {},
            Real coreStdDevs = Null<Real>(),
            Real coreFraction = 0.5);

        //! registers options to be solved together with their expiry
        void declare(const Date& maturity,
                     const std::vector<Option::Type>& types,
                     const std::vector<Real>& strikes,
                     Exercise::Type exerciseType = Exercise::American);

        void calculate() const override;
        void update() override;

        //! strip solves run so far
        Size solveCount() const { return solveCount_; }
        //! local-vol points overwritten in the last strip solve
        Size illegalLocalVolCount() const { return illegalLocalVolCount_; }

      private:
        typedef std::pair<Date, Exercise::Type> GroupKey;
        typedef std::pair<Option::Type, Real> OptionKey;
        struct Group {
            std::vector<OptionKey> options;
            std::map<OptionKey, Size> index;
            bool valid = false;
            FdmBlackScholesStripResults results;
        };

        void add(Group& group, const OptionKey& option) const;
        void solve(const GroupKey& key, Group& group) const;

        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        DividendSchedule dividends_;
        Size xGrid_, tGrid_;
        Real minStepsPerYear_, mesherScaleFactor_, mesherEps_;
        Real spotConcentrationDensity_, illegalLocalVolOverwrite_;
        std::vector<Time> stoppingTimes_;
        Real coreStdDevs_, coreFraction_;
        mutable std::map<GroupKey, Group> groups_;
        mutable Size solveCount_ = 0;
        mutable Size illegalLocalVolCount_ = 0;
    };

}

#endif
