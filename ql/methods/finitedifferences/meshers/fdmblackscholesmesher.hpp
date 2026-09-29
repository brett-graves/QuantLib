/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2009 Klaus Spanderen

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

/*! \file fdmblackscholesmesher.hpp
    \brief 1-d mesher for the Black-Scholes process (in ln(S))
*/

#ifndef quantlib_fdm_black_scholes_mesher_hpp
#define quantlib_fdm_black_scholes_mesher_hpp

#include <ql/instruments/dividendschedule.hpp>
#include <ql/methods/finitedifferences/meshers/fdm1dmesher.hpp>

#include <ql/handle.hpp>
#include <ql/quote.hpp>

namespace QuantLib {

    class FdmQuantoHelper;
    class YieldTermStructure;
    class GeneralizedBlackScholesProcess;

    class FdmBlackScholesMesher : public Fdm1dMesher {
      public:
        FdmBlackScholesMesher(
            Size size,
            const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
            Time maturity, Real strike,
            Real xMinConstraint = Null<Real>(),
            Real xMaxConstraint = Null<Real>(),
            Real eps = 0.0001,
            Real scaleFactor = 1.5,
            const std::pair<Real, Real>& cPoint = { Null<Real>(), Null<Real>() },
            const DividendSchedule& dividendSchedule = {},
            const ext::shared_ptr<FdmQuantoHelper>& fdmQuantoHelper = {},
            Real spotAdjustment = 0.0);

        //! The [xMin, xMax] range in ln(S) the constructor spans for the
        //! same arguments: the forward's min/max over the dividend path,
        //! widened by sigma(maturity, strike) sqrt(maturity)
        //! N^{-1}(1 - eps) scaleFactor on each side, then overridden by
        //! the constraints when given.  Lets callers build a different
        //! 1-d mesher (e.g. several concentration points) over the same
        //! range.
        static std::pair<Real, Real> xRange(
            const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
            Time maturity, Real strike,
            Real xMinConstraint = Null<Real>(),
            Real xMaxConstraint = Null<Real>(),
            Real eps = 0.0001,
            Real scaleFactor = 1.5,
            const DividendSchedule& dividendSchedule = {},
            const ext::shared_ptr<FdmQuantoHelper>& fdmQuantoHelper = {},
            Real spotAdjustment = 0.0);

        static ext::shared_ptr<GeneralizedBlackScholesProcess> processHelper(
             const Handle<Quote>& s0,
             const Handle<YieldTermStructure>& rTS,
             const Handle<YieldTermStructure>& qTS,
             Volatility vol);
    };
}

#endif
