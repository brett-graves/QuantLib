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
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/pricingengines/bond/deliveryoptionlineskernel.hpp>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(DeliveryOptionLinesKernelTests)

namespace {

    const Real zMax = 6.0;
    const Size panels = 16;

    std::vector<Real> uniformEdges() {
        std::vector<Real> edges(panels + 1);
        for (Size e = 0; e <= panels; ++e)
            edges[e] = -zMax + 2.0 * zMax * e / panels;
        return edges;
    }

    Real truncation() {
        CumulativeNormalDistribution phi;
        return phi(zMax) - phi(-zMax);
    }

    // Clean price c - shift on [-1, 1] for every bond: a linear table with unit
    // negative slope and zero curvature, so the secant lines are exact.
    struct LinearBasket {
        std::vector<Real> beta, factor1, factor2, conversion, adjustment, shift;
        std::vector<Real> lower, prices, slopes, curvatures;
        std::vector<Size> length;

        LinearBasket(Real c, std::vector<Real> f1, std::vector<Real> adjust)
        : factor1(std::move(f1)), adjustment(std::move(adjust)) {
            const Size n = factor1.size();
            beta.assign(n, 1.0);
            factor2.assign(n, 0.0);
            conversion.assign(n, 1.0);
            shift.assign(n, 0.0);
            lower.assign(n, -1.0);
            length.assign(n, 2001);
            for (Size i = 0; i < n; ++i)
                for (Size k = 0; k < 2001; ++k) {
                    prices.push_back(c - (-1.0 + 0.001 * k));
                    slopes.push_back(-1.0);
                    curvatures.push_back(0.0);
                }
        }

        DeliveryOptionLinesKernel kernel() const {
            return {beta,   factor1, factor2, conversion,    adjustment,
                    shift,  lower,   0.001,   length,        prices,
                    slopes, curvatures, uniformEdges(), {0.0}, {1.0 / truncation()},
                    1e-9};
        }
    };

}

BOOST_AUTO_TEST_CASE(testSingleLinearBondIsItsMean) {
    BOOST_TEST_MESSAGE("Testing the lines kernel on one linear bond...");

    // C = 100 - (L + 0.01 z1) + 0.25: the truncated z1 is symmetric, so
    // E[C | L] = 100.25 - L.
    LinearBasket basket(100.0, {0.01}, {0.25});
    auto kernel = basket.kernel();
    kernel.evaluate({-0.02, 0.0, 0.03}, true);

    const std::vector<Real> expected = {100.27, 100.25, 100.22};
    for (Size row = 0; row < 3; ++row) {
        QL_CHECK_CLOSE_FRACTION(kernel.values()[row], expected[row], 1e-14);
        QL_CHECK_SMALL(kernel.ctdProbabilities()[row] - 1.0, 1e-14);
        QL_CHECK_SMALL(kernel.sensitivities()[row] + 1.0, 1e-14);
    }
}

BOOST_AUTO_TEST_CASE(testCrossingLinesHaveTheClosedFormMinimum) {
    BOOST_TEST_MESSAGE("Testing the lines kernel on two crossing bonds...");

    // C_A = 100 - 0.01 z1 and C_B = 100 + 0.01 z1, so min = 100 - 0.01 |z1|,
    // E[|z1| ; |z1| < 6] = 2 (phi(0) - phi(6)), each bond is CTD on half the
    // box, and each CTD-weighted derivative is -1/2.
    LinearBasket basket(100.0, {0.01, -0.01}, {0.0, 0.0});
    auto kernel = basket.kernel();
    kernel.evaluate({0.0}, true);

    NormalDistribution density;
    const Real meanAbs = 2.0 * (density(0.0) - density(zMax)) / truncation();
    QL_CHECK_CLOSE_FRACTION(kernel.values()[0], 100.0 - 0.01 * meanAbs, 1e-14);
    for (Size i = 0; i < 2; ++i) {
        QL_CHECK_SMALL(kernel.ctdProbabilities()[i] - 0.5, 1e-14);
        QL_CHECK_SMALL(kernel.sensitivities()[i] + 0.5, 1e-14);
    }
}

BOOST_AUTO_TEST_CASE(testDominatedBondIsNeverCheapest) {
    BOOST_TEST_MESSAGE("Testing that the lines kernel screens out a dearer parallel bond...");

    // The same line twice, the second 0.5 dearer, plus an exact tie: the dear
    // bond never wins and the tie goes to the earlier bond.
    LinearBasket basket(100.0, {0.01, 0.01, 0.01}, {0.0, 0.5, 0.0});
    auto kernel = basket.kernel();
    kernel.evaluate({0.01}, false);

    QL_CHECK_CLOSE_FRACTION(kernel.values()[0], 99.99, 1e-14);
    QL_CHECK_SMALL(kernel.ctdProbabilities()[0] - 1.0, 1e-14);
    QL_CHECK_SMALL(kernel.ctdProbabilities()[1], 1e-15);
    QL_CHECK_SMALL(kernel.ctdProbabilities()[2], 1e-15);
    BOOST_CHECK(kernel.sensitivities().empty());
}

BOOST_AUTO_TEST_CASE(testInconsistentInputsThrow) {
    BOOST_TEST_MESSAGE("Testing the lines kernel's input checks...");

    LinearBasket basket(100.0, {0.01, -0.01}, {0.0, 0.0});
    auto beta = basket.beta;
    beta.pop_back();
    BOOST_CHECK_THROW(DeliveryOptionLinesKernel(beta, basket.factor1, basket.factor2,
                                                basket.conversion, basket.adjustment,
                                                basket.shift, basket.lower, 0.001, basket.length,
                                                basket.prices, basket.slopes, basket.curvatures,
                                                uniformEdges(), {0.0}, {1.0}, 1e-9),
                      Error);
    auto edges = uniformEdges();
    edges[3] += 0.1;
    BOOST_CHECK_THROW(DeliveryOptionLinesKernel(basket.beta, basket.factor1, basket.factor2,
                                                basket.conversion, basket.adjustment,
                                                basket.shift, basket.lower, 0.001, basket.length,
                                                basket.prices, basket.slopes, basket.curvatures,
                                                edges, {0.0}, {1.0}, 1e-9),
                      Error);
    auto kernel = basket.kernel();
    BOOST_CHECK_THROW(kernel.evaluate({std::numeric_limits<Real>::quiet_NaN()}, false), Error);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
