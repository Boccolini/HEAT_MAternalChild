from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


UF_CODES: tuple[str, ...] = (
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
)

DATASUS_FTP_HOST = "ftp.datasus.gov.br"
SIM_DORES_DIR = "/dissemin/publicos/SIM/CID10/DORES"
SINASC_DNRES_DIR = "/dissemin/publicos/SINASC/NOV/DNRES"
SIH_RD_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados"
DATASUS_POPSVS_DIR = "/dissemin/publicos/IBGE/POPSVS"

INMET_HISTORICAL_URL_TEMPLATE = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
SINASC_OPEN_DATA_URL_TEMPLATE = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_{year}_csv.zip"
IBGE_SIDRA_6407_URL_TEMPLATE = "https://apisidra.ibge.gov.br/values/t/6407/n6/all/v/606/p/{year}/c2/4,5,6794/c58/all/f/c"

IBGE_POPULATION_TOTAL_YEAR_URLS: dict[int, str] = {
    2020: "ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2020/POP2020_20220905.xls",
    2021: "ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2021/POP2021_20240624.xls",
    2022: "ftp://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Previa_da_Populacao/POP2022_Municipios_20230622.xls",
    2023: "ftp://ftp.ibge.gov.br/Informacoes_Gerais_e_Referencia/Relacao_da_Populacao_dos_Municipios_para_publicacao_no_DOU_em_2023/POP_DOU_2023_Municipios_POP2022_Malha2023.xls",
    2024: "ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2024/POP2024_20241230.xls",
}
IBGE_POPULATION_YEAR_URLS = IBGE_POPULATION_TOTAL_YEAR_URLS


@dataclass(frozen=True)
class DownloadSpec:
    source_name: str
    dataset: str
    year: int
    url: str
    target_relative_path: str


def inmet_spec(year: int) -> DownloadSpec:
    return DownloadSpec(
        source_name="inmet",
        dataset="climate",
        year=year,
        url=INMET_HISTORICAL_URL_TEMPLATE.format(year=year),
        target_relative_path=str(PurePosixPath("data/raw/climate/inmet") / f"{year}.zip"),
    )


def sinasc_open_data_spec(year: int) -> DownloadSpec:
    return DownloadSpec(
        source_name="dados_abertos_sus",
        dataset="sinasc",
        year=year,
        url=SINASC_OPEN_DATA_URL_TEMPLATE.format(year=year),
        target_relative_path=str(PurePosixPath("data/raw/sinasc") / str(year) / f"SINASC_{year}_csv.zip"),
    )


def ibge_spec(year: int) -> DownloadSpec:
    return DownloadSpec(
        source_name="ibge_total_xls",
        dataset="population",
        year=year,
        url=IBGE_POPULATION_TOTAL_YEAR_URLS[year],
        target_relative_path=str(PurePosixPath("data/raw/population/ibge") / str(year) / PurePosixPath(IBGE_POPULATION_TOTAL_YEAR_URLS[year]).name),
    )


def ibge_sidra_population_spec(year: int) -> DownloadSpec:
    return DownloadSpec(
        source_name="ibge_sidra_6407",
        dataset="population",
        year=year,
        url=IBGE_SIDRA_6407_URL_TEMPLATE.format(year=year),
        target_relative_path=str(
            PurePosixPath("data/raw/population/ibge_sidra_6407")
            / str(year)
            / f"sidra_6407_population_municipality_age_sex_{year}.json"
        ),
    )


def datasus_popsvs_spec(year: int) -> DownloadSpec:
    year_suffix = year % 100
    filename = f"POPSBR{year_suffix:02d}.zip"
    return DownloadSpec(
        source_name="datasus_popsvs",
        dataset="population",
        year=year,
        url=f"ftp://{DATASUS_FTP_HOST}{DATASUS_POPSVS_DIR}/{filename}",
        target_relative_path=str(PurePosixPath("data/raw/population/datasus_popsvs") / str(year) / filename),
    )
