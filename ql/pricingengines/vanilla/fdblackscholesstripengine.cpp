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

#include <ql/pricingengines/vanilla/fdblackscholesstripengine.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace QuantLib {

    FdBlackScholesStripEngine::FdBlackScholesStripEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process,
        DividendSchedule dividends,
        Size xGrid,
        Size tGrid,
        Real minStepsPerYear,
        Real mesherScaleFactor,
        Real mesherEps,
        Real spotConcentrationDensity,
        Real illegalLocalVolOverwrite,
        std::vector<Time> stoppingTimes,
        Real coreStdDevs,
        Real coreFraction)
    : process_(std::move(process)), dividends_(std::move(dividends)),
      xGrid_(xGrid), tGrid_(tGrid), minStepsPerYear_(minStepsPerYear),
      mesherScaleFactor_(mesherScaleFactor), mesherEps_(mesherEps),
      spotConcentrationDensity_(spotConcentrationDensity),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite),
      stoppingTimes_(std::move(stoppingTimes)),
      coreStdDevs_(coreStdDevs), coreFraction_(coreFraction) {
        QL_REQUIRE(process_, "null process");
        registerWith(process_);
    }

    void FdBlackScholesStripEngine::declare(
        const Date& maturity,
        const std::vector<Option::Type>& types,
        const std::vector<Real>& strikes,
        Exercise::Type exerciseType) {
        QL_REQUIRE(types.size() == strikes.size(),
                   types.size() << " option types for "
                   << strikes.size() << " strikes");
        QL_REQUIRE(exerciseType == Exercise::American
                   || exerciseType == Exercise::European,
                   "American or European exercise required");
        Group& group = groups_[GroupKey(maturity, exerciseType)];
        for (Size i=0; i < types.size(); ++i)
            add(group, OptionKey(types[i], strikes[i]));
    }

    void FdBlackScholesStripEngine::add(Group& group,
                                        const OptionKey& option) const {
        if (group.index.count(option) != 0)
            return;
        group.index[option] = group.options.size();
        group.options.push_back(option);
        group.valid = false;
    }

    void FdBlackScholesStripEngine::update() {
        for (auto& g : groups_)
            g.second.valid = false;
        VanillaOption::engine::update();
    }

    void FdBlackScholesStripEngine::solve(const GroupKey& key,
                                          Group& group) const {
        const Time maturity = process_->time(key.first);
        std::vector<ext::shared_ptr<StrikedTypePayoff> > payoffs;
        std::vector<Real> strikes;
        for (const auto& o : group.options) {
            payoffs.push_back(ext::make_shared<PlainVanillaPayoff>(o.first, o.second));
            strikes.push_back(o.second);
        }
        const Size tGrid = std::max<Size>(
            tGrid_, static_cast<Size>(std::lround(minStepsPerYear_ * maturity)));
        const ext::shared_ptr<Fdm1dMesher> mesher =
            FdmBlackScholesStripSolver::stripMesher(
                process_, maturity, strikes, dividends_, xGrid_,
                mesherScaleFactor_, mesherEps_, spotConcentrationDensity_,
                coreStdDevs_, coreFraction_);
        const FdmBlackScholesStripSolver solver(
            process_, key.first, payoffs, key.second == Exercise::American,
            dividends_, mesher, tGrid, illegalLocalVolOverwrite_, stoppingTimes_,
            true);
        group.results = solver.solve();
        group.valid = true;
        ++solveCount_;
        illegalLocalVolCount_ = group.results.illegalLocalVolCount;
    }

    void FdBlackScholesStripEngine::calculate() const {
        const ext::shared_ptr<PlainVanillaPayoff> payoff =
            ext::dynamic_pointer_cast<PlainVanillaPayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "plain vanilla payoff required");
        const Exercise::Type exerciseType = arguments_.exercise->type();
        QL_REQUIRE(exerciseType == Exercise::American
                   || exerciseType == Exercise::European,
                   "American or European exercise required");

        const GroupKey key(arguments_.exercise->lastDate(), exerciseType);
        const OptionKey option(payoff->optionType(), payoff->strike());
        Group& group = groups_[key];
        add(group, option);
        if (!group.valid)
            solve(key, group);

        const Size j = group.index.at(option);
        results_.value = group.results.value[j];
        results_.delta = group.results.delta[j];
        results_.gamma = group.results.gamma[j];
        results_.theta = group.results.theta[j];
    }

}
