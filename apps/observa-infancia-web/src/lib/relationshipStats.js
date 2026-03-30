const LANCZOS_COEFFICIENTS = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
];
function logGamma(value) {
    if (value < 0.5) {
        return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
    }
    let x = 0.9999999999998099;
    const shifted = value - 1;
    for (let index = 0; index < LANCZOS_COEFFICIENTS.length; index += 1) {
        x += LANCZOS_COEFFICIENTS[index] / (shifted + index + 1);
    }
    const t = shifted + LANCZOS_COEFFICIENTS.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
}
function betacf(a, b, x) {
    const maxIterations = 200;
    const epsilon = 3e-14;
    const fpMin = 1e-300;
    let qab = a + b;
    let qap = a + 1;
    let qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < fpMin) {
        d = fpMin;
    }
    d = 1 / d;
    let h = d;
    for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
        const even = 2 * iteration;
        let aa = (iteration * (b - iteration) * x) / ((qam + even) * (a + even));
        d = 1 + aa * d;
        if (Math.abs(d) < fpMin) {
            d = fpMin;
        }
        c = 1 + aa / c;
        if (Math.abs(c) < fpMin) {
            c = fpMin;
        }
        d = 1 / d;
        h *= d * c;
        aa = (-(a + iteration) * (qab + iteration) * x) / ((a + even) * (qap + even));
        d = 1 + aa * d;
        if (Math.abs(d) < fpMin) {
            d = fpMin;
        }
        c = 1 + aa / c;
        if (Math.abs(c) < fpMin) {
            c = fpMin;
        }
        d = 1 / d;
        const delta = d * c;
        h *= delta;
        if (Math.abs(delta - 1) < epsilon) {
            break;
        }
    }
    return h;
}
function regularizedIncompleteBeta(x, a, b) {
    if (x <= 0) {
        return 0;
    }
    if (x >= 1) {
        return 1;
    }
    const betaTerm = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) {
        return (betaTerm * betacf(a, b, x)) / a;
    }
    return 1 - (betaTerm * betacf(b, a, 1 - x)) / b;
}
function twoTailedPValueFromT(tStatistic, degreesOfFreedom) {
    if (!Number.isFinite(tStatistic) || degreesOfFreedom <= 0) {
        return 0;
    }
    const absoluteT = Math.abs(tStatistic);
    const x = degreesOfFreedom / (degreesOfFreedom + absoluteT * absoluteT);
    const pValue = regularizedIncompleteBeta(x, degreesOfFreedom / 2, 0.5);
    return Math.min(1, Math.max(0, pValue));
}
function significanceLabel(pValue) {
    if (pValue < 0.001) {
        return "Significant at the 0.1% level";
    }
    if (pValue < 0.01) {
        return "Significant at the 1% level";
    }
    if (pValue < 0.05) {
        return "Significant at the 5% level";
    }
    if (pValue < 0.1) {
        return "Weak evidence below the 10% level";
    }
    return "Not significant at the 10% level";
}
export function formatPValue(pValue) {
    if (pValue === null || !Number.isFinite(pValue)) {
        return "NA";
    }
    if (pValue < 0.001) {
        return "<0.001";
    }
    return pValue.toFixed(3);
}
export function computeLinearRegression(points) {
    const valid = points.filter((point) => Number.isFinite(point.x_value) && Number.isFinite(point.y_value));
    if (valid.length < 3) {
        return null;
    }
    const n = valid.length;
    const meanX = valid.reduce((sum, point) => sum + point.x_value, 0) / n;
    const meanY = valid.reduce((sum, point) => sum + point.y_value, 0) / n;
    const sumSquaresX = valid.reduce((sum, point) => sum + (point.x_value - meanX) ** 2, 0);
    const sumSquaresY = valid.reduce((sum, point) => sum + (point.y_value - meanY) ** 2, 0);
    if (!Number.isFinite(sumSquaresX) || sumSquaresX <= 0) {
        return null;
    }
    const covariance = valid.reduce((sum, point) => sum + (point.x_value - meanX) * (point.y_value - meanY), 0);
    const slope = covariance / sumSquaresX;
    const intercept = meanY - slope * meanX;
    const residualSumSquares = valid.reduce((sum, point) => {
        const fitted = intercept + slope * point.x_value;
        return sum + (point.y_value - fitted) ** 2;
    }, 0);
    const rSquared = sumSquaresY > 0 ? Math.max(0, Math.min(1, 1 - residualSumSquares / sumSquaresY)) : 0;
    const degreesOfFreedom = n - 2;
    let tStatistic = Number.NaN;
    let pValue = Number.NaN;
    if (degreesOfFreedom > 0) {
        if (residualSumSquares === 0) {
            tStatistic = Number.POSITIVE_INFINITY;
            pValue = 0;
        }
        else {
            const residualVariance = residualSumSquares / degreesOfFreedom;
            const standardErrorSlope = Math.sqrt(residualVariance / sumSquaresX);
            if (standardErrorSlope === 0) {
                tStatistic = Number.POSITIVE_INFINITY;
                pValue = 0;
            }
            else {
                tStatistic = slope / standardErrorSlope;
                pValue = twoTailedPValueFromT(tStatistic, degreesOfFreedom);
            }
        }
    }
    const minX = Math.min(...valid.map((point) => point.x_value));
    const maxX = Math.max(...valid.map((point) => point.x_value));
    return {
        n,
        slope,
        intercept,
        rSquared,
        tStatistic,
        pValue,
        significanceLabel: significanceLabel(pValue),
        significantAtFivePercent: Number.isFinite(pValue) && pValue < 0.05,
        line: [
            [minX, intercept + slope * minX],
            [maxX, intercept + slope * maxX],
        ],
    };
}
export function computeLinearRegressionByStratum(points) {
    return computeLinearRegressionByGroup(points, (point) => point.context_stratum ?? "missing");
}
export function computeLinearRegressionByGroup(points, getGroupKey) {
    const grouped = new Map();
    for (const point of points) {
        const stratum = getGroupKey(point);
        const bucket = grouped.get(stratum) ?? [];
        bucket.push(point);
        grouped.set(stratum, bucket);
    }
    const output = [];
    for (const [stratum, stratumPoints] of grouped.entries()) {
        const regression = computeLinearRegression(stratumPoints);
        if (!regression) {
            continue;
        }
        output.push({ stratum, regression });
    }
    return output;
}
