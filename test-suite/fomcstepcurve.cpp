/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Brett Graves

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

#include "toplevelfixture.hpp"
#include "utilities.hpp"
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/termstructures/yield/fomcstepcurve.hpp>
#include <ql/termstructures/yield/oisratehelper.hpp>
#include <ql/termstructures/yield/overnightindexfutureratehelper.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(FomcStepCurveTests)

namespace {

    const Date ref(14, October, 2026);

    std::vector<Date> decisions() {
        return {Date(28, October, 2026), Date(9, December, 2026), Date(27, January, 2027),
                Date(17, March, 2027),   Date(28, April, 2027),   Date(9, June, 2027)};
    }

    FomcStepCurveConfig smallConfig() {
        FomcStepCurveConfig c;
        c.horizonYears = 0.75;
        c.splineKnotsYears = {2, 4, 7};
        c.curveYears = 10;
        c.ultimateForwardBp = 420.0;  // the truth's flat tail: no penalty at the truth
        return c;
    }

    ext::shared_ptr<FomcStepParametrisation> param(bool withEffr = false) {
        return ext::make_shared<FomcStepParametrisation>(
            ref, decisions(), UnitedStates(UnitedStates::SOFR), smallConfig(), withEffr);
    }

    Size bdayIndex(const FomcStepParametrisation& p, const Date& d) {
        const std::vector<Date>& b = p.businessDays();
        auto it = std::find(b.begin(), b.end(), d);
        QL_REQUIRE(it != b.end(), d << " is not a business day of the parametrisation");
        return it - b.begin();
    }

    //! level 390, equal moves of 5bp, turns at their priors, flat spline: no penalty
    Array truth(const FomcStepParametrisation& p) {
        Array x = p.initialGuess(390.0);
        Size nMoves = p.movesRange().second - p.movesRange().first;
        for (Size k = p.movesRange().first; k < p.movesRange().second; ++k)
            x[k] = 5.0;
        for (Size j = p.splineRange().first; j < p.splineRange().second; ++j)
            x[j] = 390.0 + 5.0 * nMoves;
        return x;
    }

    struct Targets {
        std::vector<ext::shared_ptr<RateHelper> > helpers;
        std::vector<ext::shared_ptr<SimpleQuote> > quotes;
        std::vector<std::string> names;
        std::vector<Real> bp, hw;
        std::vector<bool> effr, synthetic;
    };

    //! monthly arithmetic SOFR averages through 18 months, then OIS par rates
    Targets targets() {
        Targets t;
        auto sofr = ext::make_shared<Sofr>();
        Date start = ref;
        for (Size m = 0; m < 18; ++m) {
            Date end = Date::endOfMonth(start) + 1;
            auto q = ext::make_shared<SimpleQuote>(100.0);
            t.helpers.push_back(ext::make_shared<OvernightIndexFutureRateHelper>(
                Handle<Quote>(q), start, end, sofr, Handle<Quote>(), RateAveraging::Simple));
            t.quotes.push_back(q);
            t.names.push_back("m" + std::to_string(m));
            t.bp.push_back(-0.01);
            start = end;
        }
        for (Integer y : {2, 3, 4, 5, 7, 9}) {
            auto q = ext::make_shared<SimpleQuote>(0.04);
            t.helpers.push_back(
                ext::make_shared<OISRateHelper>(2, Period(y, Years), Handle<Quote>(q), sofr));
            t.quotes.push_back(q);
            t.names.push_back("ois" + std::to_string(y));
            t.bp.push_back(1e-4);
        }
        t.hw.assign(t.helpers.size(), 0.25);
        t.effr.assign(t.helpers.size(), false);
        t.synthetic.assign(t.helpers.size(), false);
        return t;
    }

    FomcStepFit makeFit(const ext::shared_ptr<FomcStepParametrisation>& p, const Targets& t) {
        return FomcStepFit(p, t.helpers, t.names, t.bp, t.hw, t.effr, t.synthetic,
                           Actual365Fixed());
    }

    //! quote every target at its value on the curve shown at x
    void quoteAt(FomcStepFit& fit, const Array& x, const Targets& t) {
        fit.show(x);
        for (Size i = 0; i < t.helpers.size(); ++i)
            t.quotes[i]->setValue(t.helpers[i]->impliedQuote());
    }

}

BOOST_AUTO_TEST_CASE(testMovesStepOnTheBusinessDayAfterEachDecision) {
    BOOST_TEST_MESSAGE("Testing that FOMC moves take effect the business day after the decision...");
    Settings::instance().evaluationDate() = ref;
    auto p = param();
    Array x(p->size(), 0.0);
    x[0] = 400.0;
    x[p->movesRange().first] = 25.0;  // the 28 October decision only
    std::vector<Real> r = p->rates(x);
    QL_CHECK_CLOSE(r[bdayIndex(*p, Date(28, October, 2026))] * 1e4, 400.0, 1e-10);
    QL_CHECK_CLOSE(r[bdayIndex(*p, Date(29, October, 2026))] * 1e4, 425.0, 1e-10);
    BOOST_CHECK(p->meetingEffectiveDates().front() == Date(29, October, 2026));
}

BOOST_AUTO_TEST_CASE(testTurnsLandOnTheLastBusinessDayOfTheMonth) {
    BOOST_TEST_MESSAGE("Testing month-, quarter- and year-end turn placement...");
    Settings::instance().evaluationDate() = ref;
    auto p = param();
    Array x(p->size(), 0.0);
    x[p->turnsRange().first + 1] = 4.0;   // quarter-end
    x[p->turnsRange().first + 2] = 10.0;  // year-end
    std::vector<Real> r = p->rates(x);
    BOOST_CHECK_SMALL(r[bdayIndex(*p, Date(30, December, 2026))] * 1e4, 1e-10);
    QL_CHECK_CLOSE(r[bdayIndex(*p, Date(31, December, 2026))] * 1e4, 10.0, 1e-10);
    QL_CHECK_CLOSE(r[bdayIndex(*p, Date(4, January, 2027))] * 1e4, 10.0, 1e-10);
    QL_CHECK_CLOSE(r[bdayIndex(*p, Date(5, January, 2027))] * 1e4, 5.0, 1e-10);
    QL_CHECK_CLOSE(r[bdayIndex(*p, Date(31, March, 2027))] * 1e4, 4.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(testNodeCurveReproducesTheParametrisedRates) {
    BOOST_TEST_MESSAGE("Testing that the node curve's overnight forwards are the rates...");
    Settings::instance().evaluationDate() = ref;
    auto p = param();
    Targets t = targets();
    FomcStepFit fit = makeFit(p, t);
    Array x = truth(*p);
    fit.show(x);
    std::vector<Real> r = p->rates(x);
    for (Size i : {0, 5, 40, 200, 700}) {
        Date d = p->businessDays()[i];
        Real fwd = fit.sofrCurve()->forwardRate(d, d + 1, Actual360(), Simple).rate();
        BOOST_CHECK_SMALL(fwd - r[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(testFitRecoversTheParametersThatGeneratedTheQuotes) {
    BOOST_TEST_MESSAGE("Testing that the fit recovers known parameters...");
    Settings::instance().evaluationDate() = ref;
    auto p = param();
    Targets t = targets();
    FomcStepFit fit = makeFit(p, t);
    quoteAt(fit, truth(*p), t);
    fit.fit(p->initialGuess(380.0));
    for (Size i = 0; i < t.helpers.size(); ++i)
        BOOST_CHECK_SMALL(fit.residualsBp()[i], 0.01);
    QL_CHECK_CLOSE(fit.parameters()[0], 390.0, 0.02);
    BOOST_CHECK_SMALL(fit.parameters()[p->movesRange().first] - 5.0, 0.1);
    BOOST_CHECK_SMALL(fit.parameters()[p->movesRange().first + 1] - 5.0, 0.1);
}

BOOST_AUTO_TEST_CASE(testQuoteSensitivityMatchesARefit) {
    BOOST_TEST_MESSAGE("Testing dx/dq against a quadratic refit...");
    Settings::instance().evaluationDate() = ref;
    auto p = param();
    Targets t = targets();
    FomcStepFit fit = makeFit(p, t);
    quoteAt(fit, truth(*p), t);
    fit.fit(p->initialGuess(380.0), "quadratic");
    Array base = fit.parameters();
    Matrix dxdq = fit.dxdq();
    Size j = 1;  // November average: pins the first meeting
    t.quotes[j]->setValue(t.quotes[j]->value() - 0.01);  // rate up 1bp
    fit.fit(base, "quadratic", false, true);
    Size i = p->movesRange().first;
    Real refit = fit.parameters()[i] - base[i];
    BOOST_CHECK(std::fabs(refit) > 0.3);
    BOOST_CHECK_SMALL(refit - dxdq[i][j], 0.02);
}

BOOST_AUTO_TEST_CASE(testAWiderBandLetsTheCurveLeaveThatQuote) {
    BOOST_TEST_MESSAGE("Testing that widening a target's band lowers its weight...");
    Settings::instance().evaluationDate() = ref;
    auto p = param();
    Targets t = targets();
    FomcStepFit fit = makeFit(p, t);
    quoteAt(fit, truth(*p), t);
    Size j = 1;  // November average: pins the first meeting
    t.quotes[j]->setValue(t.quotes[j]->value() - 0.03);  // 3bp off the others
    fit.fit(p->initialGuess(380.0), "quadratic");
    Real tight = std::fabs(fit.residualsBp()[j]);
    Array base = fit.parameters();

    std::vector<Real> hw = t.hw;
    hw[j] *= 10.0;
    fit.setHalfWidthsBp(hw);
    BOOST_CHECK_EQUAL(fit.halfWidthsBp()[j], hw[j]);
    fit.fit(base, "quadratic", false, true);
    Real wide = std::fabs(fit.residualsBp()[j]);
    BOOST_CHECK(wide > tight + 0.5);

    fit.setHalfWidthsBp(t.hw);  // back to the original bands: the original fit
    fit.fit(fit.parameters(), "quadratic", false, true);
    BOOST_CHECK_SMALL(std::fabs(fit.residualsBp()[j]) - tight, 1e-4);  // solver tolerance

    BOOST_CHECK_THROW(fit.setHalfWidthsBp(std::vector<Real>(hw.size() - 1, 0.25)), Error);
    hw[0] = 0.0;
    BOOST_CHECK_THROW(fit.setHalfWidthsBp(hw), Error);
}

BOOST_AUTO_TEST_CASE(testFixedJumpsShiftTheSofrRatesOnly) {
    BOOST_TEST_MESSAGE("Testing fixed jumps on the SOFR overnight path...");
    Settings::instance().evaluationDate() = ref;
    FomcStepCurveConfig cfg = smallConfig();
    auto plain = ext::make_shared<FomcStepParametrisation>(ref, decisions(), UnitedStates(UnitedStates::SOFR), cfg, true);
    const std::vector<Date>& b = plain->businessDays();
    Date monthEnd = UnitedStates(UnitedStates::SOFR).endOfMonth(Date(30, November, 2026));
    Size iMe = bdayIndex(*plain, monthEnd);
    cfg.jumpDates = {monthEnd, b[iMe + 1]};
    cfg.jumpBp = {4.0, 2.5};
    auto jumped = ext::make_shared<FomcStepParametrisation>(ref, decisions(), UnitedStates(UnitedStates::SOFR), cfg, true);
    Array x = truth(*plain);
    std::vector<Real> r0 = plain->rates(x, false), r1 = jumped->rates(x, false);
    for (Size i = 0; i < r0.size(); ++i) {
        Real expected = i == iMe ? 4e-4 : i == iMe + 1 ? 2.5e-4 : 0.0;
        BOOST_CHECK_SMALL(r1[i] - r0[i] - expected, 1e-15);
    }
    std::vector<Real> e0 = plain->rates(x, true), e1 = jumped->rates(x, true);
    for (Size i = 0; i < e0.size(); ++i)
        BOOST_CHECK_SMALL(e1[i] - e0[i], 1e-15);
    const std::vector<Date>& nodes = jumped->nodeDates();
    BOOST_CHECK(std::find(nodes.begin(), nodes.end(), monthEnd) != nodes.end());

    cfg.jumpDates = {Date(1, October, 2026), monthEnd};  // a past date is dropped, as on a rolled curve
    cfg.jumpBp = {9.0, 4.0};
    FomcStepParametrisation rolled(ref, decisions(), UnitedStates(UnitedStates::SOFR), cfg, true);
    BOOST_CHECK_SMALL(rolled.rates(x, false)[iMe] - r0[iMe] - 4e-4, 1e-15);

    cfg.jumpDates = {Date(28, November, 2026)};  // a Saturday
    cfg.jumpBp = {1.0};
    BOOST_CHECK_THROW(FomcStepParametrisation(ref, decisions(), UnitedStates(UnitedStates::SOFR), cfg, true),
                      Error);
}

BOOST_AUTO_TEST_CASE(testEffrLegIsThePolicyPathAndSofrAddsTheSpread) {
    BOOST_TEST_MESSAGE("Testing the EFFR leg against the SOFR rates...");
    Settings::instance().evaluationDate() = ref;
    auto p = param(true);
    Array x(p->size(), 0.0);
    x[0] = 380.0;
    x[p->movesRange().first] = 25.0;
    for (Size j = p->spreadRange().first; j < p->spreadRange().second; ++j)
        x[j] = 2.0;
    x[p->turnsRange().first + 2] = 10.0;
    std::vector<Real> sofr = p->rates(x), effr = p->rates(x, true);
    Size a = bdayIndex(*p, Date(15, October, 2026)), b = bdayIndex(*p, Date(28, October, 2026)),
         c = bdayIndex(*p, Date(29, October, 2026)), d = bdayIndex(*p, Date(31, December, 2026));
    QL_CHECK_CLOSE((sofr[a] - effr[a]) * 1e4, 2.0, 1e-9);
    QL_CHECK_CLOSE((effr[c] - effr[b]) * 1e4, 25.0, 1e-9);
    QL_CHECK_CLOSE((sofr[d] - effr[d]) * 1e4, 12.0, 1e-9);
    BOOST_CHECK(p->names()[p->spreadRange().first] == "spread:2026-10-14");
}

BOOST_AUTO_TEST_CASE(testTailIsPinnedToTheUltimateForward) {
    BOOST_TEST_MESSAGE("Testing that the far tail bends to the ultimate forward...");
    Settings::instance().evaluationDate() = ref;
    FomcStepCurveConfig cfg = smallConfig();
    cfg.curveYears = 60;
    cfg.splineKnotsYears = {2, 4, 7, 20, 40};
    cfg.ultimateForwardBp = 200.0;
    auto p = ext::make_shared<FomcStepParametrisation>(ref, decisions(), UnitedStates(UnitedStates::SOFR), cfg);
    Targets t = targets();
    FomcStepFit fit(p, t.helpers, t.names, t.bp, t.hw, t.effr, t.synthetic, Actual365Fixed());
    quoteAt(fit, truth(*p), t);  // quotes from a flat 420bp tail out to 9y
    fit.fit(p->initialGuess(380.0));
    std::vector<Real> r = p->rates(fit.parameters());
    // Instruments still fit; the forward at the far end is the ultimate forward.
    for (Size i = 0; i < t.helpers.size(); ++i)
        BOOST_CHECK_SMALL(fit.residualsBp()[i], 0.3);
    QL_CHECK_CLOSE(r.back() * 1e4, 200.0, 1.0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
