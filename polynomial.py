from __future__ import annotations

import math
from collections import defaultdict

import numpy
from numpy.typing import NDArray


class Polynomial:
    coefficients: NDArray[numpy.int64]

    def __init__(self, coefficients: NDArray[numpy.int64]) -> None:
        self.coefficients = coefficients[
            : 1 + max(i for i, c in enumerate(coefficients) if i == 0 or c != 0)
        ]

    @staticmethod
    def from_string(s: str) -> Polynomial:
        coeff: defaultdict[int, int] = defaultdict(lambda: 0)
        for x in s.strip().replace(" + ", " +").replace(" - ", " -").split(" "):
            if x != "":
                if "x" in x:
                    x1, x2 = x.split("x")
                    x2 = "^1" if "^" not in x2 else x2
                    x1 = x1 + "1*" if "*" not in x1 else x1
                    coeff[int(x2[1:])] += int(x1[:-1])
                else:
                    coeff[0] += int(x)
        order = max(coeff.keys())
        coefficients = numpy.zeros((order + 1,), numpy.int64)
        for k, v in coeff.items():
            coefficients[k] = v
        return Polynomial(coefficients)

    def __repr__(self) -> str:
        c = self.coefficients
        return (
            " ".join(
                (
                    ("" if i == c.shape[0] - 1 else "+ ")
                    if c[i] > 0
                    else ("-" if i == c.shape[0] - 1 else "- ")
                )
                + (
                    ("" if abs(c[i]) == 1 else f"{abs(c[i])}*")
                    + ("x")
                    + ("" if i == 1 else f"^{i}")
                    if i > 0
                    else f"{abs(c[i])}"
                )
                for i in range(*self.coefficients.shape)[::-1]
                if c[i] != 0
            )
            if c.shape != (1,)
            else str(c[0])
        )

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Polynomial) and numpy.array_equal(
            self.coefficients, __o.coefficients
        )

    def __add__(self, p: Polynomial) -> Polynomial:
        coefficients = numpy.zeros(
            (max(self.coefficients.shape + p.coefficients.shape),), numpy.int64
        )
        coefficients[: self.coefficients.shape[0]] += self.coefficients
        coefficients[: p.coefficients.shape[0]] += p.coefficients
        return Polynomial(coefficients)

    def __sub__(self, p: Polynomial) -> Polynomial:
        coefficients = numpy.zeros(
            (max(self.coefficients.shape + p.coefficients.shape),), numpy.int64
        )
        coefficients[: self.coefficients.shape[0]] += self.coefficients
        coefficients[: p.coefficients.shape[0]] -= p.coefficients
        return Polynomial(coefficients)

    def __mul__(self, p: Polynomial) -> Polynomial:
        return Polynomial(numpy.convolve(self.coefficients, p.coefficients))

    def __truediv__(self, p: Polynomial) -> RationalPolynomial:
        return RationalPolynomial(self, p)


class RationalPolynomial:
    num: Polynomial
    denom: Polynomial

    def __init__(self, num: Polynomial, denom: Polynomial) -> None:
        gcd = math.gcd(*num.coefficients, *denom.coefficients)
        num = Polynomial(num.coefficients // gcd)
        denom = Polynomial(denom.coefficients // gcd)

        self.num = num
        self.denom = denom

    @staticmethod
    def from_string(s: str) -> RationalPolynomial:
        s1, s2 = s.split("/")
        return RationalPolynomial(
            Polynomial.from_string(s1.strip("( )")),
            Polynomial.from_string(s2.strip("( )")),
        )

    def __repr__(self) -> str:
        return f"({self.num.__repr__()}) / ({self.denom.__repr__()})"

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, RationalPolynomial)
            and self.num == __o.num
            and self.denom == __o.denom
        )

    def __add__(self, r: RationalPolynomial) -> RationalPolynomial:
        return RationalPolynomial(
            self.num * r.denom + r.num * self.denom, self.denom * r.denom
        )

    def __sub__(self, r: RationalPolynomial) -> RationalPolynomial:
        return RationalPolynomial(
            self.num * r.denom - r.num * self.denom, self.denom * r.denom
        )

    def __mul__(self, r: RationalPolynomial) -> RationalPolynomial:
        return RationalPolynomial(self.num * r.num, self.denom * r.denom)

    def __truediv__(self, r: RationalPolynomial) -> RationalPolynomial:
        return RationalPolynomial(self.num * r.denom, self.denom * r.num)


if __name__ == "__main__":
    print(RationalPolynomial.from_string("(2*x^2 + 2*x)/(2)"))
