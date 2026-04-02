import geopandas as gpd
import pandas as pd
import numpy as np
import glob
from PIL import Image

from shapely.geometry import Point
from shapely.ops import unary_union
import torchvision.transforms as v2
from torch.utils.data import Dataset

from PIL import Image
import warnings
from .utils import *
warnings.filterwarnings("ignore")
from scipy.stats import shapiro
from .config import *
from .model_configurations import *

class PreMergerData:
    def __init__(self, full_pictures_path, countries, settings):

        self.full_pictures_path = full_pictures_path
        self.countries = countries

        self.settings = settings

        shapefile = gpd.read_file(GLOBAL_DATA_OTH_PATH+'NUTS_RG_20M_2021_4326.shp')
        shapefile.loc[shapefile['NUTS_ID']=='RO21', 'NAME_LATN'] = 'Nord-Est RO' # there is another nord-est in italy
        self.shapefile = shapefile.loc[shapefile['LEVL_CODE']==3][['NUTS_ID', 'geometry']]
        self.shapefile['NUTS_ID'] = self.shapefile['NUTS_ID'].apply(lambda x: '_'.join([x[:2], x[2], x[3], x[4]]))

        self.centroid_distance_matrix = self.shapefile.to_crs('+proj=cea').centroid.to_crs('EPSG:4326').apply(
            lambda g: self.shapefile.distance(g));
        self.touching = self.shapefile.to_crs('+proj=cea').centroid.to_crs('EPSG:4326').apply(lambda g: self.shapefile.touches(g));
        self.country = self.shapefile['NUTS_ID'].apply(lambda x: x[:2]==self.shapefile['NUTS_ID'].str[:2])
    
    def preprocess(self):
        total_c = pd.DataFrame()

        y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))
              ] for x in glob.glob(self.full_pictures_path+'/*/*.jpg')]
        
        if len(y) == 0:
            y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))
              ] for x in glob.glob(self.full_pictures_path+'/train/*/*.jpg')]

        y = gpd.GeoDataFrame(y).rename(columns={0:'path', 1:'geometry'})
        y = y.set_geometry('geometry', crs='EPSG:4326')
        y = gpd.sjoin(y, self.shapefile).drop(columns='index_right').copy()

        to_be_merged = y.groupby('NUTS_ID')['NUTS_ID'].count()

        y['add_cat2'] = np.nan
        y['add_cat2'] = y['add_cat2'].astype(str)

        merges = []

        for x in to_be_merged.loc[to_be_merged<30].index:
            ind = self.shapefile['NUTS_ID'].loc[self.shapefile['NUTS_ID']==x].index[0]
            distances = self.centroid_distance_matrix[ind].drop(ind).copy()
            touch_ar = self.touching[ind].drop(ind).copy()
            self_country_ar = self.country[ind].drop(ind).copy()

            distances += np.abs(touch_ar-1)*100000
            distances += np.abs(self_country_ar-1)*100000
            id_to_merge_to = self.shapefile['NUTS_ID'].loc[distances.loc[distances==distances.min()].index[0]]
            merges.append((id_to_merge_to, x))

        result = []

        for tup in merges:
            for idx, already in enumerate(result):
                if any(item in already for item in tup):
                    result[idx] = already + tuple(item for item in tup if item not in already)
                    break
            else:
                result.append(tup)

        for i, x in enumerate(result):
            for xx in x:
                y.loc[y['NUTS_ID']==xx, 'add_cat2'] = x[0]+str(i)

        y['NUTS_ID_ne'] = y.apply(lambda x: x['add_cat2'] if x['add_cat2'] != 'nan' else x['NUTS_ID'], axis=1)
        y = y.drop(columns = ['add_cat2'])

        total_c = y.copy()

        x3 = self.shapefile.join(total_c[['NUTS_ID', 'NUTS_ID_ne']].drop_duplicates().set_index('NUTS_ID'), on='NUTS_ID')
        x3['NUTS_ID_ne'] = x3['NUTS_ID_ne'].fillna(x3['NUTS_ID'])
        self.premerged_shapes = x3.groupby('NUTS_ID_ne')['geometry'].apply(lambda x: unary_union(x))
        self.premerged_shapes.crs = "EPSG:4326"
        self.premerged_shapes = gpd.GeoDataFrame(self.premerged_shapes, crs='EPSG:4326')

class DataContainer:
    def __init__(self, polygons, full_pictures_path, countries, conditions):

        self.polygons = polygons
        self.full_pictures_path = full_pictures_path
        self.countries = countries
        self.conditions = conditions
        self.explicite_train_test = False
    
    def load_pictures(self):

        y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))
              ] for x in glob.glob(self.full_pictures_path+'/*/*.jpg')]
        
        if len(y) == 0: # meaning that there must be split into train and test explicite
            y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))
              ] for x in glob.glob(self.full_pictures_path+'/train/*/*.jpg')]

            y = gpd.GeoDataFrame(y).rename(columns={0:'path', 1:'geometry'})
            y = y.set_geometry('geometry', crs='EPSG:4326')
            y_train = gpd.sjoin_nearest(y, self.polygons)
            y_train['test_ind'] = 0

            y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))
              ] for x in glob.glob(self.full_pictures_path+'/test/*/*.jpg')]
            
            y = gpd.GeoDataFrame(y).rename(columns={0:'path', 1:'geometry'})
            y = y.set_geometry('geometry', crs='EPSG:4326')
            y_test = gpd.sjoin_nearest(y, self.polygons)
            y_test['test_ind'] = 1

            y = pd.concat([y_train, y_test], ignore_index=True)
            self.explicite_train_test = True

        else:
            y = gpd.GeoDataFrame(y).rename(columns={0:'path', 1:'geometry'})
            y = y.set_geometry('geometry', crs='EPSG:4326')
            y = gpd.sjoin_nearest(y, self.polygons)

        self.pictures = y.copy().rename(columns={'NUTS_ID_ne':'NUTS_ID_fin'})
        self.pictures = self.pictures.groupby(level=0).first()
    
    def apply_conditions(self):
        self.pictures = self.pictures.loc[self.pictures['path'].str[-6:-4].isin(self.conditions)]

    def precalculateHaversineDist(self):
        a = self.pictures.join(pd.Series(self.polygons.to_crs('+proj=cea').centroid.to_crs('EPSG:4326'), name='centroids'), on='NUTS_ID_fin')
        self.pictures['distance_to_centroid'] = a.apply(lambda x: haversine_distance(
            x['geometry'].x, x['geometry'].y, x['centroids'].x, x['centroids'].y), axis=1)
        
    def attach_other_variables(self):

        # Attach GDP

        nuts_GDP = gpd.read_file('DATA_OTHER/GDP_Mapped')
        nuts_GDP['centroid_loc'] = nuts_GDP['geometry'].centroid

        parts = []

        for country_code in nuts_GDP['NUTS_ID'].str[:2].unique():

            selected_country_shapes = nuts_GDP.loc[nuts_GDP['CNTR_CODE']==country_code]
            selected_country_points = self.pictures.loc[self.pictures['NUTS_ID_fin'].str[:2]==country_code]
            distance_matrix = selected_country_shapes.centroid_loc.apply(lambda g: selected_country_points.distance(g))
            normalized_weights = distance_matrix/distance_matrix.sum(axis=0)
            interpolated_GDP = selected_country_shapes['final_GDP'] @ normalized_weights
            parts.append(interpolated_GDP)

        res_norm = {}
        for tre in range(500, 1000):
            pct2 = pd.concat([self.pictures, pd.concat(parts)], axis=1)
            pct2.loc[pct2['final_GDP']>pct2['final_GDP'].quantile(tre/1000), 'final_GDP'] = pct2['final_GDP'].quantile(tre/1000)
            res_norm[tre/1000] = shapiro(pct2['final_GDP']).pvalue
        best_treshold = pd.Series(res_norm).index[pd.Series(res_norm).argmax()]

        self.GDP_upper_treshold = best_treshold

        pct2 = pd.concat([self.pictures, pd.concat(parts)], axis=1)
        pct2.loc[pct2['final_GDP']>pct2['final_GDP'].quantile(best_treshold), 'final_GDP'] = pct2['final_GDP'].quantile(best_treshold)

        # Attach METEO data

        meteo_data = gpd.read_file('DATA_OTHER/Meteo_mapped')
        meteo_data['Point'] = [Point(x.y, x.x) for x in meteo_data['geometry']]
        meteo_data = meteo_data.drop(columns=['geometry']).set_geometry('Point')

        photo_dates = gpd.read_file('DATA_OTHER/photo_dates')
        pct2 = pct2.sjoin_nearest(photo_dates.loc[~pd.isna(photo_dates['date'])])
        pct2['month'] = pct2['date'].dt.month
        pct2 = pct2.drop(columns='index_right')

        res = []
        for month in pct2['month'].unique():
            pct2_sel = pct2.loc[pct2['month']==month]
            meteo_sel = meteo_data.loc[meteo_data['month']==month].drop(columns=['month'])
            pct2_sel = pct2_sel.sjoin_nearest(meteo_sel)
            res.append(pct2_sel)

        pct3 = pd.concat(res).drop(columns=['index_right', 'id', 'month'])
        meteo_sel = meteo_data.loc[meteo_data['month']==7].drop(columns=['month'])
        pct4 = pct3.sjoin_nearest(meteo_sel, rsuffix='June', lsuffix='Exact').drop(columns=['index_June'])
        self.pictures = pct4.sort_index()

        
class DataFeederOperator:
    def __init__(self, total_images, country_selector=[], countries_all=[], explicite_train_test=False):
        self.total_images = total_images
        self.country_selector = country_selector
        self.countries_all = countries_all

        self.country_dictionary = {y:i for i, y in enumerate(self.countries_all)}
        self.country_dictionary_back = {v:k for k,v in self.country_dictionary.items()}

        self.panorama = False

        self.explicite_train_test = explicite_train_test

    def select_data(self):
        if self.country_selector != []:
            self.selected_images = self.total_images.loc[self.total_images['path'].apply(
                lambda x: x.rsplit('/', 2)[1]).isin(self.country_selector)]
        else:
            self.selected_images = self.total_images

        if self.panorama:

            self.selected_images['path'] = self.selected_images['path'].apply(lambda x: '|'.join(
                ['*' if i == 1 else x for i, x in enumerate(x.rsplit('|', 2))]))
            self.selected_images['real_path_location'] = self.selected_images['path'].str.rsplit('|', n=1).apply(lambda x: x[0])
        self.selected_images['real_path_location'] = self.selected_images['path']

    def train_test_split(self, test_perc=0.2):

        if self.explicite_train_test:
            self.test_paths=self.selected_images.loc[self.selected_images['test_ind']==1].drop_duplicates()
            self.train_paths=self.selected_images.loc[self.selected_images['test_ind']==0].drop_duplicates()

        else:

            selection_df = self.selected_images.groupby('real_path_location')['NUTS_ID_fin'].first().reset_index()

            test_paths = selection_df.groupby('NUTS_ID_fin').sample(frac=test_perc).set_index('real_path_location')
            train_paths = selection_df.set_index('real_path_location').drop(test_paths.index).sort_index()

            self.test_paths=self.selected_images.join(test_paths, on='real_path_location', how='inner', rsuffix='trash'
                                                    ).drop(columns = ['real_path_location', 'NUTS_ID_fintrash']).drop_duplicates()
            self.train_paths=self.selected_images.join(train_paths, on='real_path_location', how='inner', rsuffix='trash'
                                                    ).drop(columns = ['real_path_location', 'NUTS_ID_fintrash']).drop_duplicates()

convert_tensor = v2.ToTensor()
class GeoBrainDataset(Dataset):
    def __init__(self, img_dir, target_transform=None, transform=None, load_embeddings=False, preprocess=lambda x: x, name='noname'):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.load_embeddings = load_embeddings
        self.name = name

        if self.load_embeddings:
            self.target_dir = 'embeddings/%s' % self.load_embeddings
            all_embeddings = torch.load(self.target_dir + '/all_embeddings.pt', map_location='cpu')
            self.all_embeddings = {k:v.to('cpu') for k,v in all_embeddings.items() if k[-5:-3] in CONDITIONS}

            preset_embeddings = [x.rsplit('|', 3)[0] for x in all_embeddings.keys()]
            ii = len(list(all_embeddings.keys())[0].split('/'))
            self.img_dir = self.img_dir.loc[self.img_dir['path'].apply(lambda x: '|'.join('/'.join(x.rsplit('/', ii)[1:]).split('|', 2)[:2])).isin(preset_embeddings)]
        else:
            self.all_paths = set([x for x in glob.glob(GLOBAL_DATA_PATH+'/*/*.jpg')])

        self.id_translator = self.img_dir.index.values

        self.meteo_data = self.img_dir[['pr_Exact', 'ta6_Exact', 'rg_Exact', 'tn_Exact', 'tx_Exact',
                                        'ws_Exact', 'pd_Exact', 'pr6_Exact', 'pr_June',
                                        'ta6_June', 'rg_June', 'tn_June', 'tx_June', 'ws_June', 'pd_June',
                                        'pr6_June']]
        self.meteo_normalization_params = self.meteo_data.agg(['mean', 'std'])
        self.meteo_data = (self.meteo_data-self.meteo_normalization_params.loc['mean'])/self.meteo_normalization_params.loc['std']

        self.GDP_data = self.img_dir[['final_GDP']]
        self.GDP_normalization_params = self.GDP_data.agg(['mean', 'std'])
        self.GDP_data = (self.GDP_data-self.GDP_normalization_params.loc['mean'])/self.GDP_normalization_params.loc['std']

        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dir)
    
    def processCurrentItem(self, folder_path):
        images = [self.preprocess(Image.open(path)) for path in glob.glob(folder_path+'*.jpg')]
        if self.transform is not None:
            images = [self.transform(image) for image in images]

        if len(images) > 1:
            images = torch.stack(images, dim=3)#.to(DEVICE)
            force3pano = True
        else:
            images = images[0]#.to(DEVICE)
            force3pano = False

        if force3pano:
            if images.dim() == 3:
                images = torch.stack((images,images,images), dim=3)
            elif images.dim() == 4 and images.shape[3] == 2:
                images = torch.cat((images,images[:, :, :, [0]]), dim=3)
        return images, None

    def __getitem__(self, id):
        with torch.no_grad():
            idx = self.id_translator[id]

            img_path = self.img_dir.loc[idx]['path']

            if '*' in img_path:
                force3pano = True
                paths = [img_path.replace('*', x) for x in ['120', '240', '360']]
            else:
                force3pano = False
                paths = [img_path]

            if self.load_embeddings:
                paths = [x.split(system_configs[self.name]['data_location'])[1][1:].replace('jpg', 'pt') for x in paths]
                images = [self.all_embeddings[path][0, :] for path in paths if path in self.all_embeddings]

                if len(images) > 1:
                    images = torch.stack(images, dim=1)
                else:
                    images = images[0]

                if force3pano:
                    if images.dim() == 1:
                        images = torch.stack((images,images,images), dim=1)
                    elif images.dim() == 2 and images.shape[1] == 2:
                        images = torch.cat((images,images[:, [0]]), dim=1)
            
            else:
                images = [self.preprocess(Image.open(path)) for path in paths if path in self.all_paths]
                if self.transform is not None:
                    images = [self.transform(image) for image in images]

                if len(images) > 1:
                    images = torch.stack(images, dim=3)#.to(DEVICE)
                else:
                    images = images[0]#.to(DEVICE)

                if force3pano:
                    if images.dim() == 3:
                        images = torch.stack((images,images,images), dim=3)
                    elif images.dim() == 4 and images.shape[3] == 2:
                        images = torch.cat((images,images[:, :, :, [0]]), dim=3)

            d_t_c = self.img_dir.loc[idx]['distance_to_centroid']

            if self.target_transform is not None:
                idx, d_t_c  = self.target_transform(idx, d_t_c)

            oth_data_meteo = torch.Tensor(self.meteo_data.loc[idx].values)
            oth_data_gdp = torch.Tensor(self.GDP_data.loc[idx].values)

            return images, (
                            idx, d_t_c, 
                            self.img_dir.loc[idx]['geometry'].x, 
                            self.img_dir.loc[idx]['geometry'].y, 
                            self.img_dir.loc[idx]['NUTS_ID_fin'], 
                            torch.cat((oth_data_meteo, oth_data_gdp))#.to(DEVICE)
                            )
import json
class GeoBrainDatasetFractured(Dataset):
    def __init__(self, img_dir, target_transform=None, transform=None, load_embeddings=False, preprocess=lambda x: x, name='noname'):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.load_embeddings = load_embeddings
        self.name = name

        if self.load_embeddings:
            # 1. Load mapping
            mapping_path = f'embeddings/{self.load_embeddings}/mapping.json'
            with open(mapping_path, 'r') as f:
                self.path_mapping = json.load(f)

            # 2. Open the raw binary file using memmap
            mmap_path = f'embeddings/{self.load_embeddings}/all_embeddings.npy'

            num_items = len(self.path_mapping)
            self.emb_dim = 1152  # Ensure this matches your SigLIP dimension (usually 1152 for SO400M)

            # Use np.memmap instead of np.load
            self.all_embeddings = np.memmap(
                mmap_path, 
                dtype='float32', 
                mode='r', 
                shape=(num_items, self.emb_dim)
            )

            # Filtering logic (optimized for the mapping)
            # Filter keys based on CONDITIONS
            valid_keys = [k for k in self.path_mapping.keys() if k[-5:-3] in CONDITIONS]
            preset_embeddings = set([x.rsplit('|', 3)[0] for x in valid_keys])
            
            # Filter the dataframe to match existing embeddings
            ii = len(list(self.path_mapping.keys())[0].split('/'))
            self.img_dir = self.img_dir.loc[
                self.img_dir['path'].apply(
                    lambda x: '|'.join('/'.join(x.rsplit('/', ii)[1:]).split('|', 2)[:2])
                ).isin(preset_embeddings)
            ]
        else:
            self.all_paths = set([x for x in glob.glob(GLOBAL_DATA_PATH+'/*/*.jpg')])

        self.id_translator = self.img_dir.index.values

        # ... (Normalization logic remains the same) ...
        self.meteo_data = self.img_dir[['pr_Exact', 'ta6_Exact', 'rg_Exact', 'tn_Exact', 'tx_Exact',
                                        'ws_Exact', 'pd_Exact', 'pr6_Exact', 'pr_June',
                                        'ta6_June', 'rg_June', 'tn_June', 'tx_June', 'ws_June', 'pd_June',
                                        'pr6_June']]
        self.meteo_normalization_params = self.meteo_data.agg(['mean', 'std'])
        self.meteo_data = (self.meteo_data-self.meteo_normalization_params.loc['mean'])/self.meteo_normalization_params.loc['std']

        self.GDP_data = self.img_dir[['final_GDP']]
        self.GDP_normalization_params = self.GDP_data.agg(['mean', 'std'])
        self.GDP_data = (self.GDP_data-self.GDP_normalization_params.loc['mean'])/self.GDP_normalization_params.loc['std']

        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, id):
        with torch.no_grad():
            idx = self.id_translator[id]
            img_path = self.img_dir.loc[idx]['path']

            if '*' in img_path:
                force3pano = True
                paths = [img_path.replace('*', x) for x in ['120', '240', '360']]
            else:
                force3pano = False
                paths = [img_path]

            if self.load_embeddings:
                # Convert path to the key used in the mapping
                paths = [x.split(system_configs[self.name]['data_location'])[1][1:].replace('jpg', 'pt') for x in paths]
                
                images = []
                for path in paths:
                    if path in self.path_mapping:
                        # Fetch ONLY the index we need from the 30GB file
                        mmap_idx = self.path_mapping[path]
                        # .copy() ensures we have a clean RAM-based array for this sample
                        emb_np = self.all_embeddings[mmap_idx].copy()
                        images.append(torch.from_numpy(emb_np))

                if len(images) > 1:
                    images = torch.stack(images, dim=1)
                elif len(images) == 1:
                    images = images[0]
                else:
                    # Fallback for missing keys
                    images = torch.zeros(self.emb_dim)

                if force3pano:
                    if images.dim() == 1:
                        images = torch.stack((images,images,images), dim=1)
                    elif images.dim() == 2 and images.shape[1] == 2:
                        images = torch.cat((images,images[:, [0]]), dim=1)
            
            else:
                images = [self.preprocess(Image.open(path)) for path in paths if path in self.all_paths]
                if self.transform is not None:
                    images = [self.transform(image) for image in images]

                if len(images) > 1:
                    images = torch.stack(images, dim=3)#.to(DEVICE)
                else:
                    images = images[0]#.to(DEVICE)

                if force3pano:
                    if images.dim() == 3:
                        images = torch.stack((images,images,images), dim=3)
                    elif images.dim() == 4 and images.shape[3] == 2:
                        images = torch.cat((images,images[:, :, :, [0]]), dim=3)

            # ... (Rest of the metadata processing) ...
            d_t_c = self.img_dir.loc[idx]['distance_to_centroid']
            if self.target_transform is not None:
                idx, d_t_c  = self.target_transform(idx, d_t_c)

            oth_data_meteo = torch.Tensor(self.meteo_data.loc[idx].values)
            oth_data_gdp = torch.Tensor(self.GDP_data.loc[idx].values)

            return images, (idx, d_t_c, 
                            self.img_dir.loc[idx]['geometry'].x, 
                            self.img_dir.loc[idx]['geometry'].y, 
                            self.img_dir.loc[idx]['NUTS_ID_fin'], 
                            torch.cat((oth_data_meteo, oth_data_gdp)))
