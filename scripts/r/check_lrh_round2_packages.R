args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(name, default = "") {
  index <- match(name, args)
  if (is.na(index) || index == length(args)) {
    return(default)
  }
  args[[index + 1]]
}

json_array <- function(values) {
  if (length(values) == 0) {
    return("[]")
  }
  paste0("[", paste(sprintf('"%s"', values), collapse = ","), "]")
}

packages_arg <- parse_arg("--packages", "")
packages <- Filter(nzchar, trimws(strsplit(packages_arg, ",", fixed = TRUE)[[1]]))
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

status <- if (length(missing_packages) == 0) "PASS" else "FAIL"
cat(
  sprintf(
    '{"status":"%s","missing_packages":%s}\n',
    status,
    json_array(missing_packages)
  )
)

if (length(missing_packages) > 0) {
  quit(status = 1)
}
