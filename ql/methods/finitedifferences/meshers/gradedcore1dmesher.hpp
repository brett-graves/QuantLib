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

/*! \file gradedcore1dmesher.hpp
    \brief One-dimensional mesher: a uniform core with geometrically
           graded wings
*/

#ifndef quantlib_graded_core_1d_mesher_hpp
#define quantlib_graded_core_1d_mesher_hpp

#include <ql/methods/finitedifferences/meshers/fdm1dmesher.hpp>
#include <utility>

namespace QuantLib {

    //! Uniform core around a point, geometrically graded out to the ends
    /*! The core is uniform with spacing \f$ h = w / k \f$ on
        \f$ [c - w, c + w] \f$, \f$ k = (n_c - 1)/2 \f$, where \f$ n_c \f$
        is the odd node count closest to \p coreFraction times \p size;
        \p center is a node.  Core nodes outside \f$ (start, end) \f$ are
        dropped and their share goes to the wings.

        Each wing runs from the core edge to its end of the range with
        spacing \f$ h r, h r^2, \dots \f$, the ratio \f$ r \ge 1 \f$
        solved so that the last node lands on the range end; a wing
        that needs no growth is uniform.  The remaining nodes are split
        between the wings in proportion to their lengths.

        Compared with a sinh-concentrated mesh at the same size, the
        core width is set directly (e.g. in standard deviations) and
        does not shrink as the range widens, while the spacing still
        varies smoothly (no seam at the core edge).
    */
    class GradedCore1dMesher : public Fdm1dMesher {
      public:
        GradedCore1dMesher(Real start,
                           Real end,
                           Size size,
                           Real center,
                           Real coreHalfWidth,
                           Real coreFraction = 0.5);

        //! growth ratios of the (lower, upper) wing; 1 when uniform
        std::pair<Real, Real> growthRatios() const { return ratios_; }

      private:
        std::pair<Real, Real> ratios_;
    };

}

#endif
