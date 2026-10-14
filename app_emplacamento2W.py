"""
Dashboard de Análise e Previsão de Emplacamentos
------------------------------------------------
Execução:
    streamlit run app_emplacamento.py

Requisitos:
    pip install streamlit pandas numpy plotly statsmodels
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Emplacamentos | Análise e Previsão",
    page_icon="🏍️",
    layout="wide",
)

# Compatibilidade entre versões do Streamlit (width="stretch" x use_container_width)
def _versao_streamlit() -> tuple[int, int]:
    try:
        partes = st.__version__.split(".")
        return int(partes[0]), int(partes[1])
    except Exception:  # noqa: BLE001  # pragma: no cover
        return (0, 0)


_NOVO_API = _versao_streamlit() >= (1, 49)  # width="stretch" a partir da 1.49

LARGURA = {"width": "stretch"} if _NOVO_API else {"use_container_width": True}


# ---------------------------------------------------------------------------
# Configuração de colunas
# ---------------------------------------------------------------------------

TARGET = "Emplacamento"
DATE_COL = "data"

# Colunas que devem ser lidas como numéricas (decimal com vírgula)
NUMERIC_COLS = [
    "Desemprego",
    "Ipca",
    "Selic",
    "Dolar",
    "Gasolina",
    "Pib",
    "Confianca",
    "CreditoPF",
    "rendimento medio",
    "Inadimplencia",
    "Emplacamento",
]

LABELS = {
    "Desemprego": "Taxa de desemprego",
    "Ipca": "IPCA (índice)",
    "Selic": "Selic",
    "Dolar": "Dólar",
    "Gasolina": "Gasolina",
    "Pib": "PIB",
    "Confianca": "Confiança do consumidor",
    "CreditoPF": "Crédito PF",
    "rendimento medio": "Rendimento médio",
    "Inadimplencia": "Inadimplência",
    "Emplacamento": "Emplacamentos",
}

MESES_PT = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}

# Amostra embutida — permite rodar o app sem nenhum arquivo
SAMPLE_CSV = """Ano;data;Desemprego;Ipca;Selic;Dolar;Gasolina;Pib;Confianca;CreditoPF;rendimento medio;Inadimplencia;Emplacamento
2016;01/01/2016;0,096;4550,23;0,1425;3,9973;3,9973;1,5;89,04;114344;3242;4,34;91622
2016;01/02/2016;0,103;4591,18;0,1425;4,0156;4,0156;1,5;95,23;117477;3223;4,35;84200
2016;01/03/2016;0,11;4610,92;0,1425;3,5925;3,5925;1,5;89,31;127363;3238;4,3;101023
2016;01/04/2016;0,113;4639,05;0,1425;3,4352;3,4352;1,559;87,68;121045;3221;4,35;91458
2016;01/05/2016;0,113;4675,23;0,1425;3,6105;3,6105;1,559;90,85;130247;3232;4,37;81752
2016;01/06/2016;0,114;4691,59;0,1425;3,2126;3,2126;1,559;98,05;133696;3198;4,1;79522
2016;01/07/2016;0,117;4715,99;0,1425;3,2471;3,2471;1,577;97,67;125059;3200;4,15;79086
2016;01/08/2016;0,119;4736,74;0,1425;3,2266;3,2266;1,577;100,05;136820;3228;4,17;79303
2016;01/09/2016;0,119;4740,53;0,1425;3,2589;3,2589;1,577;106,95;123885;3224;4,23;68854
2016;01/10/2016;0,119;4752,86;0,1425;3,1878;3,1878;1,633;105,98;128364;3230;4,23;64077
2016;01/11/2016;0,12;4761,42;0,14;3,3827;3,671;1,633;110,3;136710;3236;4,15;72285
2016;01/12/2016;0,121;4775,7;0,1375;3,2532;3,734;1,633;110,73;144799;3252;4;82901
2017;01/01/2017;0,127;4793,85;0,1375;3,1507;3,768;1,586;102,25;129135;3264;4,04;67064
2017;01/02/2017;0,133;4809,67;0,13;3,1086;3,755;1,586;113,8;114218;3274;4,06;60041
2017;01/03/2017;0,139;4821,69;0,1225;3,123;3,687;1,586;109,38;144833;3293;4,06;82292
2017;01/04/2017;0,137;4828,44;0,1225;3,1758;3,635;1,631;109,01;123937;3278;4,09;64472
2017;01/05/2017;0,134;4843,41;0,1125;3,2262;3,617;1,631;103,49;145407;3272;4,16;78793
2017;01/06/2017;0,131;4832,27;0,1025;3,3063;3,548;1,631;100,06;144017;3260;3,91;71186
2017;01/07/2017;0,129;4843,87;0,1025;3,1259;3,553;1,649;104,78;137962;3261;4,01;69768
2017;01/08/2017;0,127;4853,07;0,0925;3,1474;3,781;1,649;101,48;147282;3256;3,99;75716
2017;01/09/2017;0,125;4860,83;0,0925;3,1614;3,881;1,649;99,67;136202;3272;3,93;65417
2017;01/10/2017;0,123;4881,25;0,0825;3,2724;3,895;1,72;102,81;146955;3283;3,93;67582
2017;01/11/2017;0,121;4894,92;0,075;3,2726;4,004;1,72;103,96;149689;3297;3,83;64634
2017;01/12/2017;0,119;4916,46;0,075;3,3121;4,085;1,72;109,46;150368;3298;3,58;76348
2018;01/01/2018;0,123;4930,72;0,07;3,1858;4,189;1,682;116,96;146870;3311;3,78;75927
2018;01/02/2018;0,127;4946,5;0,07;3,2458;4,208;1,682;120,57;131893;3324;3,76;62562
2018;01/03/2018;0,132;4950,95;0,0675;3,3046;4,199;1,682;115,64;147433;3323;3,7;78810
2018;01/04/2018;0,13;4961,84;0,065;3,5066;4,215;1,734;109,94;149286;3333;3,68;81563
2018;01/05/2018;0,128;4981,69;0,065;3,7225;4,314;1,734;113,5;154719;3334;3,64;79918
2018;01/06/2018;0,126;5044,46;0,065;3,8765;4,552;1,734;104;152422;3329;3,52;73522
2018;01/07/2018;0,124;5061,11;0,065;3,7557;4,492;1,768;103,48;154678;3318;3,56;75725
2018;01/08/2018;0,122;5056,56;0,065;4,0545;4,447;1,768;104,38;169264;3332;3,55;88097
2018;01/09/2018;0,12;5080,83;0,065;4,0477;4,625;1,768;106,79;148599;3318;3,47;73604
2018;01/10/2018;0,118;5103,69;0,065;3,7218;4,717;1,82;107,91;168137;3321;3,43;82927
2018;01/11/2018;0,117;5092,97;0,065;3,8662;4,59;1,82;114,5;176652;3324;3,35;76273
2018;01/12/2018;0,117;5100,61;0,065;3,8804;4,365;1,82;127,8;174064;3343;3,3;83320
2019;01/01/2019;0,122;5116,93;0,065;3,6439;4,268;1,758;128,64;163631;3363;3,39;90169
2019;01/02/2019;0,125;5138,93;0,065;3,7511;4,19;1,758;139,39;158806;3375;3,35;83762
2019;01/03/2019;0,128;5177,47;0,065;3,9238;4,305;1,758;125,53;159687;3361;3,41;83499
2019;01/04/2019;0,126;5206,98;0,065;3,9207;4,437;1,827;121,71;173250;3344;3,41;92978
2019;01/05/2019;0,124;5213,75;0,065;3,9218;4,552;1,827;117,01;182704;3320;3,46;97535
2019;01/06/2019;0,121;5214,27;0,065;3,8518;4,468;1,827;107,38;171635;3320;3,4;79611
2019;01/07/2019;0,119;5224,18;0,065;3,8125;4,351;1,881;110,92;190336;3312;3,49;89580
2019;01/08/2019;0,119;5229,93;0,06;4,1445;4,316;1,881;114,58;187191;3326;3,53;88131
2019;01/09/2019;0,119;5227,84;0,06;4,1551;4,326;1,881;112,19;186451;3325;3,6;87363
2019;01/10/2019;0,117;5233,07;0,055;4,0174;4,38;1,924;111,82;199718;3349;3,58;97942
2019;01/11/2019;0,113;5259,76;0,05;4,2364;4,413;1,924;118,64;194279;3364;3,59;92873
2019;01/12/2019;0,111;5320,25;0,05;4,019;4,531;1,924;121,26;212357;3353;3,56;93446
2020;01/01/2020;0,113;5331,42;0,045;4,282;4,579;1,868;121,27;190362;3361;3,63;91346
2020;01/02/2020;0,117;5344,75;0,045;4,4733;4,55;1,868;131,79;173921;3364;3,72;79554
2020;01/03/2020;0,124;5348,49;0,0425;5,2046;4,462;1,868;124,58;178336;3391;4;75046
2020;01/04/2020;0,127;5331,91;0,0375;5,4858;4,066;1,753;111,97;146895;3428;4,13;28184
2020;01/05/2020;0,131;5311,65;0,0375;5,3361;3,818;1,753;96,82;150308;3484;4,04;29042
2020;01/06/2020;0,136;5325,46;0,03;5,4661;3,964;1,753;100,43;170639;3538;3,67;45688
2020;01/07/2020;0,141;5344,63;0,0225;5,224;4,144;1,929;102,84;182540;3580;3,52;84787
2020;01/08/2020;0,148;5357,46;0,0225;5,4914;4,237;1,929;102,62;189834;3575;3,35;95597
2020;01/09/2020;0,149;5391,75;0,02;5,6112;4,297;1,929;108,36;196409;3577;3,19;99118
2020;01/10/2020;0,146;5438,12;0,02;5,7446;4,358;2,059;107,61;204528;3524;3,11;95695
2020;01/11/2020;0,143;5486,52;0,02;5,3319;4,406;2,059;111,65;214205;3487;3,02;89023
2020;01/12/2020;0,142;5560,59;0,02;5,1937;4,483;2,059;111,71;218765;3434;2,89;98318
2021;01/01/2021;0,144;5574,49;0,02;5,4625;4,622;2,157;116,09;180884;3426;2,93;85514
2021;01/02/2021;0,146;5622,43;0,02;5,5986;4,951;2,157;116,23;182157;3395;2,97;57159
2021;01/03/2021;0,149;5674,72;0,02;5,6315;5,484;2,157;112,9;210804;3404;2,92;62006
2021;01/04/2021;0,148;5692,31;0,0275;5,4366;5,448;2,204;104,95;212275;3363;2,95;94397
2021;01/05/2021;0,147;5739,56;0,0275;5,2172;5,604;2,204;105,9;231343;3360;3,01;110127
2021;01/06/2021;0,142;5769,98;0,035;4,9686;5,687;2,204;107,32;235960;3308;2,91;106445
2021;01/07/2021;0,137;5825,37;0,0425;5,2123;5,807;2,296;110,98;240061;3277;2,94;112292
2021;01/08/2021;0,132;5876,05;0,0425;5,1492;5,933;2,296;111,99;258718;3231;2,96;102392
2021;01/09/2021;0,127;5944,21;0,0525;5,4428;6,078;2,296;114,66;252946;3174;3,01;108560
2021;01/10/2021;0,121;6018,51;0,0625;5,6372;6,341;2,356;109,37;250107;3127;2,99;96754
2021;01/11/2021;0,116;6075,69;0,0775;5,6239;6,744;2,356;109,35;271405;3086;3,01;106179
2021;01/12/2021;0,111;6120,04;0,0775;5,5703;6,67;2,356;112;273348;3061;3,05;111944
2022;01/01/2022;0,112;6153,09;0,0925;5,3041;6,635;2,32;108,35;239283;3089;3,26;89436
2022;01/02/2022;0,112;6215,24;0,0925;5,1599;6,6;2,32;104,05;225349;3093;3,38;73986
2022;01/03/2022;0,111;6315,93;0,1075;4,739;7,012;2,32;105,12;275124;3106;3,45;109729
2022;01/04/2022;0,105;6382,88;0,1175;4,9721;7,245;2,517;104,35;267797;3094;3,54;107445
2022;01/05/2022;0,098;6412,88;0,1175;4,7315;7,28;2,517;105,87;290873;3113;3,65;133002
2022;01/06/2022;0,093;6455,85;0,1275;5,2562;7,25;2,517;103,6;287826;3136;3,57;120577
2022;01/07/2022;0,091;6411,95;0,1325;5,1734;6,05;2,601;105,58;291986;3180;3,71;107252
2022;01/08/2022;0,089;6388,87;0,1325;5,1831;5,39;2,601;106,76;319342;3208;3,78;117786
2022;01/09/2022;0,087;6370,34;0,1375;5,4154;5;2,601;111,46;306060;3252;3,8;122933
2022;01/10/2022;0,083;6407,93;0,1375;5,1791;4,89;2,641;114,23;306122;3271;3,89;119736
2022;01/11/2022;0,081;6434,2;0,1375;5,1851;5,04;2,641;116,79;288189;3302;3,92;122787
2022;01/12/2022;0,079;6474,09;0,1375;5,286;4,97;2,641;122,84;288643;3310;3,94;131288
2023;01/01/2023;0,084;6508,4;0,1375;5,0731;5,05;2,581;125,38;272683;3324;4,11;110320
2023;01/02/2023;0,086;6563,07;0,1375;5,2367;5,09;2,581;128,78;243678;3323;4,16;100310
2023;01/03/2023;0,088;6609,67;0,1375;5,0631;5,52;2,581;127,42;289428;3330;4,11;145519
2023;01/04/2023;0,085;6649,99;0,1375;4,9865;5,51;2,728;125,1;255622;3320;4,26;120356
2023;01/05/2023;0,083;6665,28;0,1375;5,0574;5,39;2,728;122,22;291098;3314;4,35;160843
2023;01/06/2023;0,08;6659,95;0,1375;4,786;5,38;2,728;125,27;282248;3329;4,25;139740
2023;01/07/2023;0,079;6667,94;0,1375;4,7241;5,61;2,769;124,46;291920;3341;4,22;122255
2023;01/08/2023;0,078;6683,28;0,1375;4,9544;5,71;2,769;131,02;322838;3351;4,12;141979
2023;01/09/2023;0,077;6700,66;0,1325;5,032;5,83;2,769;132,64;300826;3384;4;134463
2023;01/10/2023;0,076;6716,74;0,1275;5,035;5,73;2,865;132,75;312870;3395;3,97;137218
2023;01/11/2023;0,075;6735,55;0,1275;4,9205;5,63;2,865;129,74;317236;3424;3,88;130060
2023;01/12/2023;0,074;6773,27;0,1225;4,8521;5,6;2,865;133,46;310877;3410;3,75;132243
2024;01/01/2024;0,076;6801,72;0,1175;4,9526;5,57;2,765;133,25;314388;3448;3,76;142815
2024;01/02/2024;0,078;6858,17;0,1125;4,9716;5,75;2,765;138,22;291440;3463;3,76;135842
2024;01/03/2024;0,079;6869,14;0,1125;5,0153;5,74;2,765;132,52;305408;3460;3,71;151943
2024;01/04/2024;0,075;6895,24;0,1075;5,1934;5,8;2,935;129,59;320982;3476;3,75;169779
2024;01/05/2024;0,071;6926,96;0,1075;5,2443;5,86;2,935;126,46;329539;3497;3,82;164041
2024;01/06/2024;0,069;6941,51;0,105;5,5925;5,85;2,935;127,05;313832;3520;3,71;165434
2024;01/07/2024;0,068;6967,89;0,105;5,65;6,04;3,016;127,78;351472;3499;3,78;156327
2024;01/08/2024;0,066;6966,5;0,105;5,6103;6,11;3,016;127,27;347106;3518;3,82;163095
2024;01/09/2024;0,064;6997,15;0,105;5,4482;6,08;3,016;123,23;348030;3506;3,83;156146
2024;01/10/2024;0,062;7036,33;0,1075;5,7867;6,09;3,064;120,97;358711;3525;3,79;166262
2024;01/11/2024;0,061;7063,77;0,1075;5,973;6,1;3,064;123,93;342829;3541;3,75;146648
2024;01/12/2024;0,062;7100,5;0,1125;6,1778;6,14;3,064;125,62;358619;3555;3,63;151028
2025;01/01/2025;0,065;7111,86;0,1225;5,8415;6,18;3,025;123,51;337762;3572;3,89;151412
2025;01/02/2025;0,068;7205,03;0,1325;5,8854;6,36;3,025;120,48;328225;3586;3,96;155477
2025;01/03/2025;0,07;7245,38;0,1325;5,7055;6,34;3,025;115,21;337442;3596;4,03;165578
2025;01/04/2025;0,066;7276,54;0,1425;5,6748;6,32;3,2;111,01;349001;3585;4,25;182219
2025;01/05/2025;0,062;7295,46;0,1425;5,7227;6,28;3,2;111,73;348206;3600;4,36;192822
2025;01/06/2025;0,058;7312,97;0,1475;5,4326;6,23;3,2;112,89;350156;3635;4,41;178872
2025;01/07/2025;0,056;7331,98;0,15;5,6015;6,21;3,236;108,94;365381;3633;4,66;192727
2025;01/08/2025;0,056;7323,91;0,15;5,4304;6,19;3,236;111,88;367634;3633;4,92;184881
2025;01/09/2025;0,056;7359,06;0,15;5,3229;6,19;3,236;110,2;378569;3645;4,89;205108
2025;01/10/2025;0,054;7365,68;0,15;5,3744;6,2;3,278;113,63;393377;3661;4,99;209361
2025;01/11/2025;0,052;7378,94;0,15;5,3368;6,17;3,278;118,65;378069;3700;5,12;180132
2025;01/12/2025;0,051;7403,29;0,15;5,4778;6,19;3,278;124,46;407477;3734;5,12;191483
2026;01/01/2026;0,054;7427,72;0,15;5,2576;6,32;3,25;127,39;364403;3764;5,38;177063
2026;01/02/2026;0,058;7479,71;0,15;5,1305;6,3;3,25;127,39;337886;3774;5,56;170433
2026;01/03/2026;0,061;7545,53;0,149;5,1819;6,59;3,25;125,87;397284;3795;5,4;220768


"""


# ---------------------------------------------------------------------------
# Carga e limpeza
# ---------------------------------------------------------------------------

def _to_number(serie: pd.Series) -> pd.Series:
    """Converte texto no padrão brasileiro (1.234,56) para float."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    limpo = (
        serie.astype(str)
        .str.strip()
        .str.replace(r"[R$\s%]", "", regex=True)
        .str.replace(".", "", regex=False)   # separador de milhar
        .str.replace(",", ".", regex=False)  # separador decimal
        .replace({"": None, "nan": None, "-": None})
    )
    return pd.to_numeric(limpo, errors="coerce")


def _read_raw(conteudo: bytes | str) -> pd.DataFrame:
    """Lê CSV/TXT testando encodings e separadores comuns em exports do Excel."""
    if isinstance(conteudo, str):
        conteudo = conteudo.encode("utf-8")

    ultimo_erro: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        for sep in (None, ";", "\t", ","):
            try:
                df = pd.read_csv(
                    io.BytesIO(conteudo),
                    sep=sep,
                    encoding=enc,
                    engine="python",
                )
                if df.shape[1] >= 3:
                    return df
            except Exception as exc:  # noqa: BLE001
                ultimo_erro = exc
    raise ValueError(f"Não foi possível ler o arquivo. Último erro: {ultimo_erro}")


@st.cache_data(show_spinner=False)
def carregar(conteudo: bytes | str, nome: str = "") -> pd.DataFrame:
    if nome.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(conteudo))
    else:
        df = _read_raw(conteudo)

    df.columns = [str(c).strip() for c in df.columns]

    # Data
    col_data = next((c for c in df.columns if c.lower() in ("data", "date", "mes", "mês")), None)
    if col_data is None:
        raise ValueError("Coluna de data não encontrada.")
    df[DATE_COL] = pd.to_datetime(df[col_data], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # Numéricos
    for col in df.columns:
        if col in (DATE_COL, col_data):
            continue
        convertido = _to_number(df[col])
        if convertido.notna().mean() > 0.5:
            df[col] = convertido

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df[DATE_COL] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()
    df["Ano"] = df[DATE_COL].dt.year
    df["Mes"] = df[DATE_COL].dt.month
    df["MesNome"] = df["Mes"].map(MESES_PT)
    return df


# ---------------------------------------------------------------------------
# Modelos de previsão
# ---------------------------------------------------------------------------

def prever_ols(y: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Tendência linear + dummies mensais. Robusto para séries curtas."""
    n = len(y)
    t = np.arange(n)
    meses = y.index.month
    usar_sazonal = n >= 18 and len(np.unique(meses)) >= 6

    def design(t_arr, m_arr):
        X = [np.ones(len(t_arr)), t_arr]
        if usar_sazonal:
            for m in range(2, 13):
                X.append((m_arr == m).astype(float))
        return np.column_stack(X)

    X = design(t, meses.values)
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma = float(np.sqrt((resid**2).sum() / dof))

    fut_idx = pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS")
    Xf = design(np.arange(n, n + h), fut_idx.month.values)
    pred = Xf @ beta
    return pred, np.full(h, sigma)


def prever_holt(y: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    sazonal = "add" if len(y) >= 24 else None
    periodos = 12 if sazonal else None
    modelo = ExponentialSmoothing(
        y, trend="add", damped_trend=True,
        seasonal=sazonal, seasonal_periods=periodos,
        initialization_method="estimated",
    ).fit(optimized=True)
    pred = np.asarray(modelo.forecast(h), dtype=float)
    sigma = float(np.std(modelo.resid, ddof=1)) if len(y) > 2 else float(np.std(y))
    return pred, sigma * np.sqrt(np.arange(1, h + 1))


def prever_sarima(y: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    sazonal = (1, 1, 1, 12) if len(y) >= 30 else (0, 0, 0, 0)
    modelo = SARIMAX(
        y, order=(1, 1, 1), seasonal_order=sazonal,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    res = modelo.get_forecast(h)
    pred = np.asarray(res.predicted_mean, dtype=float)
    ic = res.conf_int(alpha=0.20)
    sigma = (np.asarray(ic.iloc[:, 1]) - pred) / 1.2816
    return pred, np.abs(sigma)


MODELOS = {
    "Tendência + Sazonalidade (OLS)": prever_ols,
    "Holt-Winters (suavização exponencial)": prever_holt,
    "SARIMA (univariado)": prever_sarima,
}

SARIMAX_NOME = "SARIMAX com exógenas defasadas"
OPCOES_MODELO = list(MODELOS) + [SARIMAX_NOME]


# --- exógenas -------------------------------------------------------------

def serie_mensal(dados: pd.DataFrame, col: str) -> pd.Series:
    s = dados.set_index(DATE_COL)[col].astype(float).asfreq("MS")
    return s.interpolate(limit_direction="both")


def projetar(s: pd.Series, idx_fut: pd.DatetimeIndex, metodo: str) -> pd.Series:
    """Projeta uma exógena para além do histórico (usado só quando lag < horizonte)."""
    if metodo.startswith("Tendência"):
        janela = s.tail(min(6, len(s)))
        passo = (janela.iloc[-1] - janela.iloc[0]) / max(len(janela) - 1, 1)
        valores = s.iloc[-1] + passo * np.arange(1, len(idx_fut) + 1)
    else:  # último valor observado
        valores = np.full(len(idx_fut), s.iloc[-1])
    return pd.Series(valores, index=idx_fut, dtype=float)


def montar_exog(
    dados: pd.DataFrame,
    vars_lags: dict[str, int],
    idx_hist: pd.DatetimeIndex,
    idx_fut: pd.DatetimeIndex,
    metodo_proj: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Monta as matrizes de exógenas defasadas para treino e previsão.

    Para um lag L e horizonte h, os L primeiros meses previstos usam valores
    JÁ OBSERVADOS. Só os meses além de L precisam de projeção.
    """
    hist, fut, projetados = {}, {}, {}
    for var, lag in vars_lags.items():
        s = serie_mensal(dados, var)
        extra = pd.date_range(s.index[-1] + pd.DateOffset(months=1),
                              periods=len(idx_fut), freq="MS")
        s_ext = pd.concat([s, projetar(s, extra, metodo_proj)])
        defasada = s_ext.shift(lag)

        nome = f"{LABELS.get(var, var)} (t-{lag})" if lag else LABELS.get(var, var)
        col_hist = defasada.reindex(idx_hist)
        if col_hist.std(skipna=True) in (0, np.nan) or not np.isfinite(col_hist.std(skipna=True)):
            continue
        hist[nome] = col_hist
        fut[nome] = defasada.reindex(idx_fut)
        projetados[nome] = max(0, len(idx_fut) - lag)

    if not hist:
        return pd.DataFrame(), pd.DataFrame(), {}

    X_hist = pd.DataFrame(hist).dropna()
    X_fut = pd.DataFrame(fut)

    # padronização (estimada no histórico) — ajuda a convergência do SARIMAX
    media, desvio = X_hist.mean(), X_hist.std().replace(0, 1)
    return (X_hist - media) / desvio, (X_fut - media) / desvio, projetados


def ajustar_sarimax(y: pd.Series, X_hist: pd.DataFrame | None, X_fut: pd.DataFrame | None, h: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if X_hist is not None and not X_hist.empty:
        idx = X_hist.index.intersection(y.index)
        y_fit, X_fit, X_prev = y.loc[idx], X_hist.loc[idx], X_fut
    else:
        y_fit, X_fit, X_prev = y, None, None

    sazonal = (1, 1, 1, 12) if len(y_fit) >= 30 else (0, 0, 0, 0)
    modelo = SARIMAX(
        y_fit, exog=X_fit, order=(1, 1, 1), seasonal_order=sazonal,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    res = modelo.get_forecast(h, exog=X_prev)
    pred = np.asarray(res.predicted_mean, dtype=float)
    ic = res.conf_int(alpha=0.20)
    sigma = np.abs((np.asarray(ic.iloc[:, 1]) - pred) / 1.2816)
    return pred, sigma, modelo


def prever_sarimax(y: pd.Series, h: int, X_hist=None, X_fut=None) -> tuple[np.ndarray, np.ndarray]:
    pred, sigma, _ = ajustar_sarimax(y, X_hist, X_fut, h)
    return pred, sigma


def tabela_lags(dados: pd.DataFrame, colunas: list[str], max_lag: int) -> pd.DataFrame:
    """Correlação de cada variável com o alvo em cada defasagem."""
    linhas = []
    for v in colunas:
        correls = {lag: dados[TARGET].corr(dados[v].shift(lag)) for lag in range(0, max_lag + 1)}
        correls = {k: x for k, x in correls.items() if pd.notna(x)}
        if not correls:
            continue
        for lag, c in correls.items():
            linhas.append({"Variável": v, "Lag": lag, "Correlação": c})
    return pd.DataFrame(linhas)


def rodar_modelo(nome: str, y: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Roda o modelo pedido; se falhar, cai para OLS e devolve o aviso."""
    try:
        pred, sigma = MODELOS[nome](y, h)
        if not np.all(np.isfinite(pred)):
            raise ValueError("previsões não finitas")
        return pred, sigma, None
    except Exception as exc:  # noqa: BLE001
        pred, sigma = prever_ols(y, h)
        return pred, sigma, f"{nome} não convergiu ({exc}). Usando tendência + sazonalidade."


def backtest(y: pd.Series, h_teste: int = 6, X_hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compara os modelos segurando as últimas observações.

    Para o SARIMAX, as exógenas do período de teste vêm dos valores reais já
    defasados — exatamente a informação que estaria disponível na prática.
    """
    if len(y) < h_teste + 8:
        return pd.DataFrame()
    treino, teste = y.iloc[:-h_teste], y.iloc[-h_teste:]
    linhas = []

    candidatos: dict[str, callable] = dict(MODELOS)
    if X_hist is not None and not X_hist.empty:
        idx_treino = X_hist.index.intersection(treino.index)
        idx_teste = X_hist.index.intersection(teste.index)
        if len(idx_teste) == h_teste and len(idx_treino) >= 8:
            candidatos[SARIMAX_NOME] = lambda s, h: prever_sarimax(
                s, h, X_hist.loc[idx_treino], X_hist.loc[idx_teste]
            )

    for nome in candidatos:
        try:
            pred, _ = candidatos[nome](treino, h_teste)
            erro = teste.values - pred
            linhas.append({
                "Modelo": nome,
                "MAPE (%)": float(np.mean(np.abs(erro / teste.values)) * 100),
                "MAE": float(np.mean(np.abs(erro))),
                "RMSE": float(np.sqrt(np.mean(erro**2))),
            })
        except Exception:  # noqa: BLE001
            continue
    if not linhas:
        return pd.DataFrame()
    return pd.DataFrame(linhas).sort_values("MAPE (%)").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configurações")

arquivo = st.sidebar.file_uploader(
    "Base de dados (CSV ou Excel)", type=["csv", "txt", "xlsx", "xls"]
)

if arquivo is not None:
    try:
        df = carregar(arquivo.getvalue(), arquivo.name)
        st.sidebar.success(f"{len(df)} linhas carregadas.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Erro na leitura: {exc}")
        st.stop()
else:
    df = carregar(SAMPLE_CSV, "amostra.csv")
    st.sidebar.info("Usando a amostra embutida. Envie sua base completa para resultados reais.")

if TARGET not in df.columns:
    st.error(f"A coluna '{TARGET}' não foi encontrada na base.")
    st.stop()

anos = sorted(df["Ano"].unique())
if len(anos) > 1:
    ini, fim = st.sidebar.select_slider(
        "Período (ano)", options=anos, value=(anos[0], anos[-1])
    )
    df = df[(df["Ano"] >= ini) & (df["Ano"] <= fim)].reset_index(drop=True)

horizonte = st.sidebar.slider("Horizonte de previsão (meses)", 1, 12, 6)
modelo_escolhido = st.sidebar.selectbox("Modelo", OPCOES_MODELO, index=len(OPCOES_MODELO) - 1)
mostrar_ic = st.sidebar.checkbox("Mostrar intervalo de confiança (80%)", value=True)

variaveis = [c for c in NUMERIC_COLS if c in df.columns and c != TARGET]

st.title("Motorcycle Registration Forecast")
st.caption(
    f"Série de {df[DATE_COL].min():%b/%Y} a {df[DATE_COL].max():%b/%Y} "
    f"· {len(df)} observações mensais"
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

serie = df.set_index(DATE_COL)[TARGET].astype(float).asfreq("MS")
serie = serie.interpolate(limit_direction="both")

ultimo = serie.iloc[-1]
anterior = serie.iloc[-2] if len(serie) > 1 else np.nan
yoy = serie.iloc[-13] if len(serie) > 12 else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Último mês", f"{ultimo:,.0f}".replace(",", "."),
          f"{(ultimo/anterior - 1)*100:+.1f}% vs. mês anterior" if pd.notna(anterior) else None)
c2.metric("Média do período", f"{serie.mean():,.0f}".replace(",", "."))
c3.metric("Acumulado 12m",
          f"{serie.tail(12).sum():,.0f}".replace(",", ".") if len(serie) >= 12 else "—")
c4.metric("Variação YoY",
          f"{(ultimo/yoy - 1)*100:+.1f}%" if pd.notna(yoy) else "—")

aba_dados, aba_insights, aba_prev = st.tabs(
    ["📄 Base de dados", "📊 Insights", "🔮 Previsão"]
)

# ---------------------------------------------------------------------------
# Aba 1 — dados
# ---------------------------------------------------------------------------

with aba_dados:
    st.subheader("Tabela completa")
    visao = df.drop(columns=["MesNome"], errors="ignore").copy()
    st.dataframe(
        visao.style.format({c: "{:,.4g}" for c in visao.select_dtypes("number").columns}),
        **LARGURA,
        height=420,
    )

    st.download_button(
        "⬇️ Baixar base tratada (CSV)",
        df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="base_tratada.csv",
        mime="text/csv",
    )

    st.subheader("Estatísticas descritivas")
    st.dataframe(
        df[[c for c in NUMERIC_COLS if c in df.columns]].describe().T.style.format("{:,.2f}"),
        **LARGURA,
    )

    st.subheader("Qualidade dos dados")
    qualidade = pd.DataFrame({
        "Tipo": df.dtypes.astype(str),
        "Nulos": df.isna().sum(),
        "% nulos": (df.isna().mean() * 100).round(1),
        "Valores únicos": df.nunique(),
    })
    st.dataframe(qualidade, **LARGURA)

    # Colunas duplicadas — na amostra, Dolar e Gasolina são idênticas
    num = df[[c for c in NUMERIC_COLS if c in df.columns]]
    duplicadas = []
    cols = list(num.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if num[a].equals(num[b]):
                duplicadas.append(f"**{a}** e **{b}**")
    if duplicadas:
        st.warning(
            "Colunas com valores idênticos: " + ", ".join(duplicadas) +
            ". Vale conferir a origem da extração antes de usá-las como variáveis distintas."
        )

    faltantes = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="MS").difference(df[DATE_COL])
    if len(faltantes):
        st.warning(f"{len(faltantes)} mês(es) ausente(s) na série: "
                   + ", ".join(d.strftime("%m/%Y") for d in faltantes[:12]))

# ---------------------------------------------------------------------------
# Aba 2 — insights
# ---------------------------------------------------------------------------

with aba_insights:
    st.subheader("Evolução dos emplacamentos")
    fig = px.line(df, x=DATE_COL, y=TARGET, markers=True)
    if len(serie) >= 12:
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.rolling(12).mean(),
            name="Média móvel 12m", line=dict(dash="dash"),
        ))
    fig.update_layout(height=420, xaxis_title="", yaxis_title="Emplacamentos",
                      hovermode="x unified")
    st.plotly_chart(fig, **LARGURA)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sazonalidade média por mês")
        sazon = df.groupby("Mes")[TARGET].mean().reindex(range(1, 13)).dropna()
        sazon.index = [MESES_PT[m] for m in sazon.index]
        fig_s = px.bar(sazon, labels={"value": "Média", "index": ""})
        fig_s.add_hline(y=df[TARGET].mean(), line_dash="dot")
        fig_s.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_s, **LARGURA)
        if len(sazon) >= 6:
            st.caption(
                f"Mês mais forte: **{sazon.idxmax()}** · mais fraco: **{sazon.idxmin()}** "
                f"(amplitude de {(sazon.max()/sazon.min() - 1)*100:.0f}%)."
            )

    with col_b:
        st.subheader("Total por ano")
        anual = df.groupby("Ano")[TARGET].sum()
        fig_a = px.bar(anual, labels={"value": "Total", "Ano": ""})
        fig_a.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_a, **LARGURA)
        if len(anual) > 1:
            var = (anual.iloc[-1] / anual.iloc[-2] - 1) * 100
            st.caption(f"{anual.index[-1]} vs. {anual.index[-2]}: **{var:+.1f}%** "
                       "(atenção se o último ano estiver incompleto).")

    st.divider()
    st.subheader("Relação com as variáveis macroeconômicas")

    if variaveis:
        corr_simples = df[variaveis + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
        fig_c = px.bar(
            corr_simples, orientation="h",
            labels={"value": "Correlação de Pearson com emplacamentos", "index": ""},
            color=corr_simples, color_continuous_scale="RdBu", range_color=[-1, 1],
        )
        fig_c.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_c, **LARGURA)

        fortes = corr_simples[corr_simples.abs() > 0.5]
        if len(fortes):
            texto = ", ".join(
                f"**{LABELS.get(k, k)}** ({v:+.2f})" for k, v in fortes.sort_values(key=abs, ascending=False).items()
            )
            st.info(f"Variáveis mais associadas aos emplacamentos: {texto}. "
                    "Correlação não implica causalidade — use como ponto de partida para a modelagem.")

        st.markdown("**Defasagem ótima (lag)** — em quantos meses a variável antecipa o emplacamento:")
        max_lag = min(12, max(1, len(df) // 3))
        linhas_lag = []
        for v in variaveis:
            melhores = {lag: df[TARGET].corr(df[v].shift(lag)) for lag in range(0, max_lag + 1)}
            melhores = {k: x for k, x in melhores.items() if pd.notna(x)}
            if not melhores:
                continue
            lag_otimo = max(melhores, key=lambda k: abs(melhores[k]))
            linhas_lag.append({
                "Variável": LABELS.get(v, v),
                "Lag (meses)": lag_otimo,
                "Correlação": round(melhores[lag_otimo], 3),
                "Correlação sem lag": round(melhores.get(0, np.nan), 3),
            })
        if linhas_lag:
            tab_lag = pd.DataFrame(linhas_lag).reindex(
                pd.DataFrame(linhas_lag)["Correlação"].abs().sort_values(ascending=False).index
            )
            st.dataframe(tab_lag, **LARGURA, hide_index=True)

        st.divider()
        st.subheader("Comparação direta")
        escolhida = st.selectbox("Variável", variaveis,
                                 format_func=lambda c: LABELS.get(c, c))
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=df[DATE_COL], y=df[TARGET], name="Emplacamentos"))
        fig_d.add_trace(go.Scatter(x=df[DATE_COL], y=df[escolhida],
                                   name=LABELS.get(escolhida, escolhida), yaxis="y2"))
        fig_d.update_layout(
            height=400, hovermode="x unified",
            yaxis=dict(title="Emplacamentos"),
            yaxis2=dict(title=LABELS.get(escolhida, escolhida), overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_d, **LARGURA)

        st.plotly_chart(
            px.scatter(df, x=escolhida, y=TARGET, trendline="ols",
                       hover_data=[DATE_COL], height=380,
                       labels={escolhida: LABELS.get(escolhida, escolhida)}),
            **LARGURA,
        )

        with st.expander("Matriz de correlação completa"):
            st.plotly_chart(
                px.imshow(df[variaveis + [TARGET]].corr(), text_auto=".2f",
                          color_continuous_scale="RdBu", zmin=-1, zmax=1, height=520),
                **LARGURA,
            )

# ---------------------------------------------------------------------------
# Aba 3 — previsão
# ---------------------------------------------------------------------------

with aba_prev:
    st.subheader(f"Previsão de {horizonte} meses a partir de {serie.index[-1]:%b/%Y}")

    if len(serie) < 6:
        st.warning("São necessárias pelo menos 6 observações para gerar previsões.")
        st.stop()

    idx_fut = pd.date_range(serie.index[-1] + pd.DateOffset(months=1),
                            periods=horizonte, freq="MS")

    X_hist = X_fut = pd.DataFrame()
    modelo_ajustado = None
    projetados: dict[str, int] = {}
    aviso = None

    if modelo_escolhido == SARIMAX_NOME:
        with st.container(border=True):
            st.markdown("**Configuração das exógenas**")

            max_lag = min(12, max(1, len(df) // 3))
            lags = tabela_lags(df, variaveis, max_lag)

            # sugestão: variáveis cuja melhor defasagem já cobre o horizonte
            sugeridas: list[str] = []
            if not lags.empty:
                elegiveis = lags[lags["Lag"] >= horizonte]
                base = elegiveis if not elegiveis.empty else lags
                ranking = (
                    base.assign(abs_corr=base["Correlação"].abs())
                    .sort_values("abs_corr", ascending=False)
                    .drop_duplicates("Variável")
                )
                sugeridas = ranking["Variável"].head(3).tolist()

            escolhidas = st.multiselect(
                "Variáveis explicativas",
                variaveis,
                default=sugeridas,
                format_func=lambda c: LABELS.get(c, c),
                help="Cada variável entra defasada. Lag ≥ horizonte significa que o "
                     "valor já é conhecido no momento da previsão.",
            )

            auto_lag = st.checkbox(
                "Escolher o lag automaticamente (maior correlação com lag ≥ horizonte)",
                value=True,
            )

            vars_lags: dict[str, int] = {}
            if escolhidas:
                colunas = st.columns(min(3, len(escolhidas)))
                for i, var in enumerate(escolhidas):
                    sub = lags[lags["Variável"] == var] if not lags.empty else pd.DataFrame()
                    candidato = sub[sub["Lag"] >= horizonte] if not sub.empty else pd.DataFrame()
                    if candidato.empty and not sub.empty:
                        candidato = sub
                    padrao = (
                        int(candidato.loc[candidato["Correlação"].abs().idxmax(), "Lag"])
                        if not candidato.empty else horizonte
                    )
                    if auto_lag:
                        vars_lags[var] = padrao
                        corr_txt = ""
                        if not sub.empty and padrao in set(sub["Lag"]):
                            c = float(sub.loc[sub["Lag"] == padrao, "Correlação"].iloc[0])
                            corr_txt = f" · correlação {c:+.2f}"
                        colunas[i % len(colunas)].caption(
                            f"**{LABELS.get(var, var)}** — lag {padrao}{corr_txt}"
                        )
                    else:
                        vars_lags[var] = colunas[i % len(colunas)].number_input(
                            LABELS.get(var, var), min_value=0, max_value=max_lag,
                            value=padrao, step=1, key=f"lag_{var}",
                        )

            metodo_proj = st.selectbox(
                "Como preencher meses sem valor observado da exógena (quando lag < horizonte)",
                ["Último valor observado", "Tendência dos últimos 6 meses"],
            )

        if vars_lags:
            X_hist, X_fut, projetados = montar_exog(
                df, vars_lags, serie.index, idx_fut, metodo_proj
            )
            if X_hist.empty:
                aviso = "Nenhuma exógena utilizável (variância nula ou dados insuficientes)."

        if X_hist.empty:
            aviso = aviso or "Sem exógenas selecionadas — rodando SARIMA univariado."

    if modelo_escolhido == SARIMAX_NOME:
        try:
            pred, sigma, modelo_ajustado = ajustar_sarimax(
                serie, X_hist if not X_hist.empty else None,
                X_fut if not X_fut.empty else None, horizonte,
            )
            if not np.all(np.isfinite(pred)):
                raise ValueError("previsões não finitas")
        except Exception as exc:  # noqa: BLE001
            pred, sigma = prever_ols(serie, horizonte)
            aviso = f"SARIMAX não convergiu ({exc}). Usando tendência + sazonalidade."
    else:
        pred, sigma, aviso = rodar_modelo(modelo_escolhido, serie, horizonte)

    if aviso:
        st.warning(aviso)

    if projetados:
        precisam = {k: v for k, v in projetados.items() if v > 0}
        if precisam:
            st.info(
                "Meses de previsão que dependem de exógena projetada: "
                + ", ".join(f"{k}: {v} de {horizonte}" for k, v in precisam.items())
                + ". Quanto maior o lag, menos projeção é necessária."
            )
        else:
            st.success(
                "Todas as exógenas cobrem o horizonte com valores já observados — "
                "nenhuma projeção auxiliar foi necessária."
            )

    pred = np.maximum(pred, 0)
    z = 1.2816  # 80%
    inferior = np.maximum(pred - z * sigma, 0)
    superior = pred + z * sigma

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=serie.index, y=serie.values,
                               name="Histórico", line=dict(width=2)))
    if mostrar_ic:
        fig_p.add_trace(go.Scatter(
            x=list(idx_fut) + list(idx_fut[::-1]),
            y=list(superior) + list(inferior[::-1]),
            fill="toself", fillcolor="rgba(214,110,60,0.15)",
            line=dict(width=0), name="IC 80%", hoverinfo="skip",
        ))
    fig_p.add_trace(go.Scatter(
        x=[serie.index[-1]] + list(idx_fut),
        y=[serie.values[-1]] + list(pred),
        name="Previsão", mode="lines+markers", line=dict(dash="dash", width=2),
    ))
    fig_p.update_layout(height=450, hovermode="x unified",
                        xaxis_title="", yaxis_title="Emplacamentos")
    st.plotly_chart(fig_p, **LARGURA)

    tabela_prev = pd.DataFrame({
        "Mês": [f"{MESES_PT[d.month]}/{d.year}" for d in idx_fut],
        "Previsão": pred.round(0),
        "Mínimo (80%)": inferior.round(0),
        "Máximo (80%)": superior.round(0),
    })
    ref = serie.tail(min(12, len(serie))).mean()
    tabela_prev["vs. média recente"] = [f"{(p/ref - 1)*100:+.1f}%" for p in pred]

    col_t, col_r = st.columns([2, 1])
    with col_t:
        st.dataframe(tabela_prev, **LARGURA, hide_index=True)
        st.download_button(
            "⬇️ Baixar previsão (CSV)",
            tabela_prev.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="previsao_emplacamento.csv",
            mime="text/csv",
        )
    with col_r:
        st.metric(f"Total previsto ({horizonte}m)",
                  f"{pred.sum():,.0f}".replace(",", "."))
        st.metric("Média mensal prevista",
                  f"{pred.mean():,.0f}".replace(",", "."),
                  f"{(pred.mean()/ref - 1)*100:+.1f}% vs. média recente")

    if modelo_ajustado is not None and not X_hist.empty:
        st.divider()
        st.subheader("Coeficientes das exógenas")
        resumo = pd.DataFrame({
            "Coeficiente": modelo_ajustado.params,
            "p-valor": modelo_ajustado.pvalues,
        })
        resumo = resumo.loc[[i for i in resumo.index if i in X_hist.columns]]
        resumo["Significante (5%)"] = np.where(resumo["p-valor"] < 0.05, "sim", "não")
        st.dataframe(resumo.style.format({"Coeficiente": "{:,.1f}", "p-valor": "{:.3f}"}),
                     **LARGURA)
        st.caption(
            "Exógenas padronizadas: o coeficiente indica a variação esperada nos "
            "emplacamentos para 1 desvio-padrão da variável. "
            f"AIC do modelo: {modelo_ajustado.aic:,.1f}."
        )

    st.divider()
    st.subheader("Validação (backtest)")
    bt = backtest(serie, min(6, max(1, len(serie) // 4)),
                  X_hist if not X_hist.empty else None)
    if bt.empty:
        st.info("Série curta demais para backtest — recomendo ao menos 14 observações.")
    else:
        st.dataframe(bt.style.format({"MAPE (%)": "{:.1f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}"}),
                     **LARGURA, hide_index=True)
        st.caption(f"Menor erro no teste: **{bt.iloc[0]['Modelo']}** "
                   f"(MAPE {bt.iloc[0]['MAPE (%)']:.1f}%). "
                   "O backtest usa apenas dados anteriores ao período testado.")

    with st.expander("Sobre os modelos e limitações"):
        st.markdown(
            """
- **Tendência + Sazonalidade (OLS)**: regressão com tendência linear e dummies mensais.
  Estável em séries curtas, mas não capta quebras de nível.
- **Holt-Winters**: suavização exponencial com tendência amortecida e componente sazonal
  aditivo (ativado a partir de 24 observações).
- **SARIMA (univariado)**: ARIMA(1,1,1) com componente sazonal (1,1,1,12) quando há 30+ observações.
- **SARIMAX com exógenas defasadas**: mesma estrutura, acrescida das variáveis macro
  defasadas e padronizadas. Com lag L, os L primeiros meses de previsão usam valores **já
  observados** da exógena; só os meses além de L dependem da projeção auxiliar. Por isso o
  lag sugerido é sempre ≥ horizonte quando existe um com correlação relevante.

O intervalo de 80% reflete apenas o erro do modelo — não captura choques regulatórios,
tributários ou de oferta, nem o erro de projeção das exógenas. Cuidado com colinearidade:
Selic, crédito PF e inadimplência caminham juntas, e empilhar as três tende a produzir
coeficientes instáveis sem ganho de acurácia. O backtest é o critério, não o AIC isolado.
            """
        )
