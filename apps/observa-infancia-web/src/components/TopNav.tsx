import { Link, NavLink } from "react-router-dom";
import type { PageDefinition } from "../lib/api";

type TopNavProps = {
  pages: PageDefinition[];
};

export function TopNav({ pages }: TopNavProps) {
  return (
    <nav className="flex flex-wrap gap-2">
      {pages
        .filter((page) => page.route !== "/methods")
        .map((page) => (
          <NavLink
            key={page.page_id}
            to={page.route}
            className={({ isActive }) =>
              [
                "rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "border-navy bg-navy text-white"
                  : "border-rule bg-white text-navy hover:border-blue hover:text-blue"
              ].join(" ")
            }
          >
            {page.title}
          </NavLink>
        ))}
      <NavLink
        to="/methods"
        className={({ isActive }) =>
          [
            "rounded-full border px-4 py-2 text-sm font-medium transition-colors",
            isActive
              ? "border-blue bg-blue text-white"
              : "border-rule bg-white text-navy hover:border-blue hover:text-blue"
          ].join(" ")
        }
      >
        Methods
      </NavLink>
      <Link
        to="/how-to-use"
        className="rounded-full border border-rule bg-white px-4 py-2 text-sm font-medium text-navy transition-colors hover:border-blue hover:text-blue"
      >
        How to use
      </Link>
      <Link
        to="/outcome-matrix"
        className="rounded-full border border-blue bg-blue/5 px-4 py-2 text-sm font-medium text-blue transition-colors hover:bg-blue hover:text-white"
      >
        32 outcomes matrix
      </Link>
      <Link
        to="/version-ledger"
        className="rounded-full border border-rule bg-white px-4 py-2 text-sm font-medium text-navy transition-colors hover:border-blue hover:text-blue"
      >
        Version ledger
      </Link>
    </nav>
  );
}
