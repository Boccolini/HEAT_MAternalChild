import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
export function useDashboardFilters(manifest) {
    const [searchParams, setSearchParams] = useSearchParams();
    const filters = useMemo(() => {
        const defaultYear = manifest?.years_covered.at(-1) ?? 2024;
        const years = (searchParams.get("years") ?? `${defaultYear}`)
            .split(",")
            .filter(Boolean)
            .map((value) => Number(value));
        return {
            page_id: searchParams.get("page") ?? undefined,
            metric_id: searchParams.get("metric") ?? undefined,
            years,
            granularity: searchParams.get("granularity") ?? "month",
            geo_level: searchParams.get("geoLevel") ?? "state",
            geo_codes: (searchParams.get("geoCodes") ?? "").split(",").filter(Boolean),
            value_mode: searchParams.get("valueMode") ?? "count",
            definition_type: searchParams.get("definitionType") ?? undefined,
            context_variable: searchParams.get("contextVariable") ?? undefined,
            context_bins: (searchParams.get("contextBins") ?? "").split(",").filter(Boolean),
            compare_mode: searchParams.get("compareMode") ??
                "none"
        };
    }, [manifest, searchParams]);
    const setFilter = (key, value) => {
        const next = new URLSearchParams(searchParams);
        if (value === undefined || value === "" || (Array.isArray(value) && value.length === 0)) {
            next.delete(key);
        }
        else if (Array.isArray(value)) {
            next.set(key, value.join(","));
        }
        else {
            next.set(key, String(value));
        }
        setSearchParams(next, { replace: true });
    };
    return { filters, setFilter };
}
