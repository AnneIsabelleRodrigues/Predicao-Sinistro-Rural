import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from PIL import Image
from io import BytesIO
import requests
import ee

class SatelliteData:
    def __init__(self):
        pass

    def save_pic(self, id_proposta, tipo, save_date, response):

        caminho_diretorio = os.path.join('D:', 'POC', 'IMAGENS DE SATÉLITE', str(id_proposta), tipo)

        if not os.path.exists(caminho_diretorio):
            os.makedirs(caminho_diretorio)

        caminho_arquivo = os.path.join(caminho_diretorio, f'{save_date}.png')

        imagem = Image.open(BytesIO(response.content))

        imagem.save(caminho_arquivo, 'PNG')

    def get_ee_feature(self, geom):

        x,y = geom.exterior.coords.xy
        coords = np.dstack((x,y)).tolist()
        g = ee.Geometry.Polygon(coords)
        return ee.Feature(g)


class Modis(SatelliteData):
    def __init__(self):
        super().__init__()

    def feat_LSTD(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        # Recupera a Imagem collection dos dados filtrados (área e data)
        image_collection = ee.ImageCollection('MODIS/061/MOD11A1')

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('LST_Day_1km')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': 13000.0,
                'max': 16500.0,
                'palette': [
                    '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
                    '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
                    '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
                    'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
                    'ff0000', 'de0101', 'c21301', 'a71001', '911003'
                ],
                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'LST_Day', save_date, response)

    def feat_NDVI(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection('MODIS/061/MOD13Q1')

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('NDVI')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': -2000,
                'max': 10000,
                'bands': ['NDVI'],
                'palette': [
                    'ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901',
                    '66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01',
                    '012e01', '011d01', '011301'
                ],

                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'NDVI', save_date, response)

    def feat_EVI(self, talhao, id_proposta, inicio_vigencia):

        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection('MODIS/061/MOD13Q1')

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('EVI')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': -2000,
                'max': 10000,
                'bands': ['EVI'],
                'palette': [
                    'ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901',
                    '66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01',
                    '012e01', '011d01', '011301'
                ],

                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'EVI', save_date, response)

    def feat_Fpar(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection("MODIS/061/MCD15A3H")

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('Fpar')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': 0.0,
                'max': 100.0,
                'bands': ['Fpar'],
                'palette': ['e1e4b4', '999d60', '2ec409', '0a4b06'],
                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'Fpar', save_date, response)

    def feat_LAI(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection("MODIS/061/MCD15A3H")

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('Lai')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': 0.0,
                'max': 100.0,
                'bands': ['Lai'],
                'palette': ['e1e4b4', '999d60', '2ec409', '0a4b06'],
                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'LAI', save_date, response)

    def feat_Gpp(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection('MODIS/061/MOD17A2H')

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('Gpp')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': 0.0,
                'max': 600.0,
                'bands': ['Gpp'],
                'palette': ['bbe029', '0a9501', '074b03'],
                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'GPP', save_date, response)

    def feat_LC(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection(self.get_ee_feature(talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection('MODIS/061/MCD12Q1')

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=1)

        s_date = start_date.strftime('%Y-%m-%d')
        e_date = past_date.strftime('%Y-%m-%d')
        save_date = date.strftime('%Y%m')

        image = ee.Image(image_collection.filterDate(s_date, e_date).select('LC_Type1').first())

        image = image.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

        download_url = image.getThumbURL({
            'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
            'dimensions': '256x256',
            'bands': ['LC_Type1'],
            'min': 1.0,
            'max': 17.0,
            'palette': [
                '05450a', '086a10', '54a708', '78d203', '009900', 'c6b044', 'dcd159',
                'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44', 'a5a5a5', 'ff6d4c',
                '69fff8', 'f9ffa4', '1c0dff'
            ],
            'format': 'png'})

        print(download_url)
        response = requests.get(download_url)

        self.save_pic(id_proposta, 'LC_Type1', save_date, response)


class Jaxa(SatelliteData):
    def __init__(self):
        super().__init__()

    def feat_Precip(self, talhao, id_proposta, inicio_vigencia):
        features = ee.FeatureCollection((self.talhao.iloc[0]['geometry']))

        image_collection = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v6/operational")

        print(type(inicio_vigencia))
        start_date = datetime.strptime(str(inicio_vigencia), '%Y-%m-%d %H:%M:%S')
        past_date = start_date - relativedelta(years=2)

        dates_months = list(pd.date_range(start=past_date, periods=24, freq='1M'))
        for date in dates_months:

            s_date = date.replace(day=1).strftime('%Y-%m-%d')
            e_date = date.strftime('%Y-%m-%d')
            save_date = date.strftime('%Y%m')

            image = image_collection.filterDate(s_date,e_date)
            image = image.select('hourlyPrecipRate')

            mediana = image.median()

            mediana = mediana.clip(self.get_ee_feature(talhao.iloc[0]['geometry']))

            download_url = mediana.getThumbURL({
                'region': self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                'dimensions': '256x256',
                'min': 0.0,
                'max': 10.0,
                'bands': ['hourlyPrecipRate'],
                'palette': ['1621a2', 'ffffff', '03ffff', '13ff03', 'efff00', 'ffb103', 'ff2300'],
                'format': 'png'})

            print(download_url)
            response = requests.get(download_url)

            self.save_pic(id_proposta, 'Precip', save_date, response)

class Histogram(SatelliteData):
    def __init__(self, ano):
        super().__init__()
        self.ano = ano

    def MapBiomas(self, talhao):

        image = ee.Image('projects/mapbiomas-workspace/public/collection8/mapbiomas_collection80_integration_v1')

        filtered_image = image.select(f'classification_{self.ano}')


        histogram = filtered_image.reduceRegion(**{'reducer':ee.Reducer.frequencyHistogram(),
                                           'geometry':self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                                           'scale':30,
                                           'maxPixels': 1e13})


        return(histogram.getInfo())

    def ModisLC_type(self, talhao):

        image_collection = ee.ImageCollection('MODIS/061/MCD12Q1')


        image = image_collection.first()
        # # Selecionar a banda desejada
        filtered_image = image.select('LC_Type1')


        histogram = filtered_image.reduceRegion(**{'reducer':ee.Reducer.frequencyHistogram(),
                                                   'geometry':self.get_ee_feature(talhao.iloc[0]['geometry']).geometry(),
                                                   'scale':30,
                                                   'maxPixels': 1e13})


        return(histogram.getInfo())








