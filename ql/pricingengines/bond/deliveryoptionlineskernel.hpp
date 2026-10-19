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

/*! \file deliveryoptionlineskernel.hpp
    \brief Conditional expectation of the cheapest-to-deliver converted price
           by closed-form lines on the main spread factor
*/

#ifndef quantlib_delivery_option_lines_kernel_hpp
#define quantlib_delivery_option_lines_kernel_hpp

#include <ql/types.hpp>
#include <vector>

namespace QuantLib {

    //! Conditional delivery-option kernel of a Treasury futures basket
    /*! Multi-factor quality-option model (Huggins & Schaller, Box 7.2):
        bond \f$ i \f$'s yield change to delivery is
        \f[ D_i = \beta_i L + a_i z_1 + b_i z_2, \f]
        with \f$ L \f$ the level change and \f$ z_1, z_2 \f$ independent
        standard normals. For each requested level the kernel returns
        \f[ g(L) = E\left[\min_i C_i \mid L\right], \qquad
            C_i = \frac{P_i(D_i) + m_i}{CF_i}, \f]
        each bond's conditional CTD probability, and optionally the
        CTD-weighted derivative
        \f$ E[1\{i \text{ is CTD}\} (P_i'(D_i) + s_i) \mid L] / CF_i \f$.

        \f$ P_i \f$ is the clean price as a function of the total shift,
        read from a uniform cubic Hermite table of values, slopes and
        curvatures; \f$ m_i \f$ is the mean adjustment and \f$ s_i \f$ the
        derivative shift supplied by the caller.

        The second factor is integrated by the caller's quadrature rule.
        For every node of it the first factor is split into uniform panels;
        on each panel every bond is replaced by its secant line in
        \f$ z_1 \f$, lowered by one twelfth of its second difference across
        the panel (the mean of the second differences at the panel's two
        edges, one-sided at the ends). The minimum of a set of lines has a
        closed-form normal expectation, so only \f$ z_2 \f$ is quadrature.
        A bond enters a panel's minimum only if its corrected panel minimum
        does not exceed the smallest panel maximum by more than the tie
        tolerance; equal slopes keep the lower line, and ties keep the
        earlier bond.

        This is the C++ port of chloride's
        \c delivery_option._conditional_lines; on chloride's frozen
        references the two agree to rounding.

        \warning One instance holds the last evaluation's results; use one
                 instance per thread.
    */
    class DeliveryOptionLinesKernel {
      public:
        /*! \param beta              level loading per bond
            \param factor1           loading on the closed-form factor \f$ z_1 \f$
            \param factor2           loading on the quadrature factor \f$ z_2 \f$
            \param conversion        conversion factor per bond, positive
            \param adjustment        mean adjustment \f$ m_i \f$ per bond
            \param derivativeShift   \f$ s_i \f$ per bond, used by the
                                     sensitivities only
            \param gridLower         total shift of each bond's first table point
            \param gridStep          uniform table step, shared by every bond
            \param gridLength        table points per bond, at least 2
            \param prices            clean prices of all tables, bond after bond
            \param slopes            first derivatives, same layout
            \param curvatures        second derivatives, same layout
            \param factor1Edges      uniform panel edges of \f$ z_1 \f$, at least 2
            \param factor2Nodes      quadrature nodes of \f$ z_2 \f$
            \param factor2Weights    their weights, already normalised for the
                                     truncation of both factors
            \param tieTolerance      screen tolerance on the panel bounds
        */
        DeliveryOptionLinesKernel(std::vector<Real> beta,
                                  std::vector<Real> factor1,
                                  std::vector<Real> factor2,
                                  std::vector<Real> conversion,
                                  std::vector<Real> adjustment,
                                  std::vector<Real> derivativeShift,
                                  std::vector<Real> gridLower,
                                  Real gridStep,
                                  const std::vector<Size>& gridLength,
                                  std::vector<Real> prices,
                                  std::vector<Real> slopes,
                                  std::vector<Real> curvatures,
                                  std::vector<Real> factor1Edges,
                                  std::vector<Real> factor2Nodes,
                                  std::vector<Real> factor2Weights,
                                  Real tieTolerance);

        //! evaluates the kernel at every level, in the given order
        void evaluate(const std::vector<Real>& levels, bool withSensitivities);

        Size bonds() const { return beta_.size(); }
        //! \f$ g(L) \f$ of the last evaluation, one per level
        const std::vector<Real>& values() const { return values_; }
        //! CTD probabilities of the last evaluation, levels by bonds, row-major
        const std::vector<Real>& ctdProbabilities() const { return ctd_; }
        //! CTD-weighted derivatives of the last evaluation, levels by bonds,
        //! row-major; empty unless sensitivities were requested
        const std::vector<Real>& sensitivities() const { return sensitivities_; }

      private:
        void evaluateLevel(Real level, bool withSensitivities, Size row);
        void tableAt(Size bond, Real shift, Size& index, Real& t) const;
        Real convertedPrice(Size bond, Real shift) const;

        std::vector<Real> beta_, factor1_, factor2_, conversion_, adjustment_, derivativeShift_;
        std::vector<Real> gridLower_;
        Real gridStep_, inverseStep_;
        std::vector<Size> gridOffset_, gridLength_;
        std::vector<Real> prices_, slopes_, curvatures_;
        std::vector<Real> edges_, factor2Nodes_, factor2Weights_;
        Real width_, tieTolerance_;
        // panel-edge normal cdf and pdf, shared by single-candidate panels
        std::vector<Real> edgeCdf_, edgePdf_;
        // per-cell scratch
        std::vector<Real> edgeValues_, edgeDerivatives_, secondDifferences_;
        std::vector<Real> alpha_, slope_, panelMin_;
        std::vector<Size> candidates_;
        std::vector<Real> bound_, correctionBound_;
        std::vector<Size> active_;
        // results
        std::vector<Real> values_, ctd_, sensitivities_;
    };

}

#endif
