import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Point
import geopandas as gpd
from unidecode import unidecode
from Processing.constants import municipios_amostra
from Processing.gee_initialize import GoogleEarthEngine
from satelliteClass import Modis, Jaxa


def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'S' or direction == 'W':
        dd *= -1
    return dd

def tratamento_dados(psrd, mun):

    psrd.replace('-', np.NaN, inplace=True)
    psrd = psrd[psrd['NM_CULTURA_GLOBAL'] == 'Soja']
    psrd = psrd[psrd['NM_CLASSIF_PRODUTO'] == 'PRODUTIVIDADE']

    psrd['sinistro_t'] = psrd['EVENTO_PREPONDERANTE'].apply(lambda x: 0 if pd.isnull(x) else 1)

    ufs = mun[mun['nome_municipio'].isin(municipios_amostra)]['nome_uf'].unique()
    ufs_trat = ["AREA_IMOVEL-" + unidecode(uf).upper().replace(" ", "")  for uf in ufs ]

    psrd['COMPLETE_LONGITUDE'] = psrd['NR_GRAU_LONG'].astype(str) + "°" + psrd['NR_MIN_LONG'].astype(str)  + "'" + psrd['NR_SEG_LONG'].astype(str) + "'' " + psrd['LONGITUDE'].astype(str)

    psrd['COMPLETE_LATITUDE'] = psrd['NR_GRAU_LAT'].astype(str) + "°" + psrd['NR_MIN_LAT'].astype(str)  + "'" + psrd['NR_SEG_LAT'].astype(str) + "'' " + psrd['LATITUDE'].astype(str)

    psrd['long'] = psrd.apply(lambda row : dms2dd(row['NR_GRAU_LONG'],row['NR_MIN_LONG'], row['NR_SEG_LONG'], row['LONGITUDE']), axis = 1)
    psrd['lat'] = psrd.apply(lambda row : dms2dd(row['NR_GRAU_LAT'],row['NR_MIN_LAT'], row['NR_SEG_LAT'], row['LATITUDE']), axis = 1)

    psrd['geometry'] = [Point(xy) for xy in zip(psrd.long, psrd.lat)]

    return psrd, ufs_trat


if __name__ == '__main__':
    psrd = pd.read_excel('files/psrdadosabertos2016a2021excel.xlsx')

    mun = pd.read_csv('../files/5 - municipio.csv', sep=';')

    psrd, ufs_trat = tratamento_dados(psrd, mun)

    psrd_data = psrd[psrd['NM_MUNICIPIO_PROPRIEDADE'].isin(municipios_amostra)]
    gdf = geopandas.GeoDataFrame(psrd_data, crs="EPSG:4326")
    points = gpd.GeoDataFrame(gdf, geometry='geometry')
    points.crs = {'init': 'epsg:4326'}

    geeclass = GoogleEarthEngine()
    geeclass.initilize()

    modis = Modis()
    jaxa = Jaxa()

    for folder in ufs_trat:

        df = gpd.read_file(f'D:/POC/ÁREA IMÓVEL - BRASIL/{folder}/AREA_IMOVEL_1.shp')
        within_points = gpd.sjoin(left_df=points, right_df=df, how='inner')
        for index, location in within_points.iterrows():
            cod_imovel = location['cod_imovel']
            talhao = df[df['cod_imovel'] == cod_imovel]
            modis.feat_LSTD(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
            modis.feat_NDVI(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
            modis.feat_EVI(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
            modis.feat_Fpar(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
            modis.feat_LAI(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
            modis.feat_Gpp(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
            jaxa.feat_Precip(talhao, location['ID_PROPOSTA'], location['DT_INICIO_VIGENCIA'])
