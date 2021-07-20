\echo Use "CREATE EXTENSION pgsklearn" to load this file. \quit

create or replace function show_logo() returns void as $$
from pgsklearn.io import read_sixel_image
plpy.info(read_sixel_image('/home/postgres/images/PolarDB-logo.sixel'))
$$ language plpython3u;
comment on function show_logo is '在终端上展示PolarDB的logo图片';

create or replace function show_lmplot(_table text, _x text, _y text, _hue text = null, _col text = null, _row text = null, _height int = 10, _aspect int = 1, _scatter boolean = true, _fit_reg boolean = true) returns void as $$
from pgsklearn.sql import select_from
from pgsklearn.plot import lm
from pandas import DataFrame

columns = [_x, _y]
if _hue is not None:
    columns.append(_hue)
if _col is not None:
    columns.append(_col)
if _row is not None:
    columns.append(_row)

records = plpy.execute(select_from(*columns, table=_table))
data = DataFrame()
for column in columns:
    data[column] = [record[column] for record in records]

plpy.info(lm(data, _x, _y, _hue, _col, _row, _height, _aspect, _scatter, _fit_reg))
$$ language plpython3u;

create or replace function show_lineplot(_table text, _x text, _y text, _hue text = null) returns void as $$
from pgsklearn.sql import select_from
from pgsklearn.plot import line
from pandas import DataFrame

columns = [_x, _y]
if _hue is not None:
    columns.append(_hue)

records = plpy.execute(select_from(*columns, table=_table))
data = DataFrame()
for column in columns:
    data[column] = [record[column] for record in records]

plpy.info(line(data, _x, _y, _hue))
$$ language plpython3u;

create or replace function show_scatterplot(_table text, _x text, _y text, _hue text = null) returns void as $$
from pgsklearn.sql import select_from
from pgsklearn.plot import scatter
from pandas import DataFrame

columns = [_x, _y]
if _hue is not None:
    columns.append(_hue)

records = plpy.execute(select_from(*columns, table=_table))
data = DataFrame()
for column in columns:
    data[column] = [record[column] for record in records]

plpy.info(scatter(data, _x, _y, _hue))
$$ language plpython3u;

create or replace function load_boston() returns void as $$
from pgsklearn.sql import create_table, insert_into, coercion
from pgsklearn.io import read_csv_records

table_name = 'boston_house_prices'
table_description = '波士顿房价数据集'
definitions = [
    {'name': 'CRIM', 'type': 'float8', 'comment': '城镇人均犯罪率'},
    {'name': 'ZN', 'type': 'float8', 'comment': '面积超25000平方英尺的住宅用地比例'},
    {'name': 'INDUS', 'type': 'float8', 'comment': '城镇中非商用地比例'},
    {'name': 'CHAS', 'type': 'float8', 'comment': '查尔斯河虚拟变量'},
    {'name': 'NOX', 'type': 'float8', 'comment': '一氧化氮浓度'},
    {'name': 'RM', 'type': 'float8', 'comment': '每栋住宅平均房间数'},
    {'name': 'AGE', 'type': 'float8', 'comment': '1940年以前建造的自住单位比例'},
    {'name': 'DIS', 'type': 'float8', 'comment': '到五个波士顿就业中心的加权距离'},
    {'name': 'RAD', 'type': 'float8', 'comment': '距离高速公路的便利指数'},
    {'name': 'TAX', 'type': 'float8', 'comment': '每一万美元的不动产税率'},
    {'name': 'PTRATIO', 'type': 'float8', 'comment': '城镇师生比例'},
    {'name': 'B', 'type': 'float8', 'comment': '城镇黑人比例'},
    {'name': 'LSTAT', 'type': 'float8', 'comment': '收入较低人口比例'},
    {'name': 'target', 'type': 'float8', 'comment': '自住房屋房价的中位数'}
]

plpy.execute(create_table(table_name, table_description, definitions))
types = [definition['type'] for definition in definitions]
sql = plpy.prepare(insert_into(table_name, definitions), types)
NR = 0
for record in coercion(definitions, read_csv_records('/home/postgres/datasets/boston.csv')):
    plpy.execute(sql, record)
    NR += 1
plpy.info(f'共导入{NR}条记录至{table_name}')
$$ language plpython3u;
comment on function load_boston is '加载波士顿房价数据集';

create or replace function load_diabetes() returns void as $$
from pgsklearn.sql import create_table, insert_into, coercion
from pgsklearn.io import read_csv_records

table_name = 'diabetes'
table_description = '糖尿病数据集'
definitions = [
    {'name': 'age', 'type': 'int', 'comment': '年龄'},
    {'name': 'sex', 'type': 'int', 'comment': '性别'},
    {'name': 'bmi', 'type': 'float8', 'comment': '体重指数'},
    {'name': 'bp', 'type': 'float8', 'comment': '血压'},
    {'name': 'tc', 'type': 'float8', 'comment': '总胆固醇'},
    {'name': 'ldl', 'type': 'float8', 'comment': '低密度脂蛋白'},
    {'name': 'hdl', 'type': 'float8', 'comment': '高密度脂蛋白'},
    {'name': 'tch', 'type': 'float8', 'comment': '总胆固醇 / 高密度脂蛋白'},
    {'name': 'ltc', 'type': 'float8', 'comment': '三脂水平'},
    {'name': 'glu', 'type': 'float8', 'comment': '血糖水平'},
    {'name': 'target', 'type': 'int', 'comment': '目标结果'}
]

plpy.execute(create_table(table_name, table_description, definitions))
types = [definition['type'] for definition in definitions]
sql = plpy.prepare(insert_into(table_name, definitions), types)
NR = 0
for record in coercion(definitions, read_csv_records('/home/postgres/datasets/diabetes.csv')):
    plpy.execute(sql, record)
    NR += 1
plpy.info(f'共导入{NR}条记录至{table_name}')
$$ language plpython3u;
comment on function load_diabetes is '加载糖尿病数据集';

create or replace function load_digits() returns void as $$
from pgsklearn.sql import create_table, insert_into, coercion
from pgsklearn.io import read_csv_records

table_name = 'digits'
table_description = '手写数字数据集'
definitions = [{'name': f'r{r}c{c}', 'type': 'int', 'comment': f'第{r}行第{c}列像素'} for r in range(1, 9) for c in range(1, 9)]
definitions.append({'name': 'target', 'type': 'int', 'comment': '目标数字'})

plpy.execute(create_table(table_name, table_description, definitions))
types = [definition['type'] for definition in definitions]
sql = plpy.prepare(insert_into(table_name, definitions), types)
NR = 0
for record in coercion(definitions, read_csv_records('/home/postgres/datasets/digits.csv')):
    plpy.execute(sql, record)
    NR += 1
plpy.info(f'共导入{NR}条记录至{table_name}')
$$ language plpython3u;
comment on function load_digits is '加载手写数字数据集';

create or replace function load_iris() returns void as $$
from pgsklearn.sql import create_table, insert_into, coercion
from pgsklearn.io import read_csv_records

table_name = 'iris'
table_description = '鸢尾花卉数据集'
definitions = [
    {'name': 'sepal_length', 'type': 'float8', 'comment': '花萼长度'},
    {'name': 'sepal_width', 'type': 'float8', 'comment': '花萼宽度'},
    {'name': 'petal_length', 'type': 'float8', 'comment': '花瓣长度'},
    {'name': 'petal_width', 'type': 'float8', 'comment': '花瓣宽度'},
    {'name': 'target', 'type': 'int', 'comment': '花卉属种 0:山鸢尾 1:变色鸢尾 2:维吉尼亚鸢尾'}
]

plpy.execute(create_table(table_name, table_description, definitions))
types = [definition['type'] for definition in definitions]
sql = plpy.prepare(insert_into(table_name, definitions), types)
NR = 0
for record in coercion(definitions, read_csv_records('/home/postgres/datasets/iris.csv')):
    plpy.execute(sql, record)
    NR += 1
plpy.info(f'共导入{NR}条记录至{table_name}')
$$ language plpython3u;
comment on function load_iris is '加载鸢尾花卉数据集';

create or replace function load_shares() returns void as $$
from pgsklearn.sql import create_table, insert_into, coercion
from pgsklearn.io import read_csv_records

table_name = 'shares'
table_description = '苹果股价数据集'
definitions = [
    {'name': 'date', 'type': 'date', 'comment': '日期'},
    {'name': 'opening_price', 'type': 'float8', 'comment': '开盘价'},
    {'name': 'closing_price', 'type': 'float8', 'comment': '收盘价'},
    {'name': 'max', 'type': 'float8', 'comment': '最高价'},
    {'name': 'min', 'type': 'float8', 'comment': '最低价'},
    {'name': 'count', 'type': 'bigint', 'comment': '总次数'},
    {'name': 'amount', 'type': 'float8', 'comment': '总金额'}
]

plpy.execute(create_table(table_name, table_description, definitions))
types = [definition['type'] for definition in definitions]
sql = plpy.prepare(insert_into(table_name, definitions), types)
NR = 0
for record in coercion(definitions, read_csv_records('/home/postgres/datasets/shares.csv')):
    plpy.execute(sql, record)
    NR += 1
plpy.info(f'共导入{NR}条记录至{table_name}')
$$ language plpython3u;
comment on function load_shares is '加载苹果股价数据集';

create or replace function decomposition_pca(table_name text, unique_key text, x_columns text[], y_columns text[]) returns void as $$
'''
PCA降维算法

Args:
    table_name: 保存训练样本与结果的表名。
    unique_key: 字段名，需要能唯一标识样本。
    x_columns: 训练数据的字段名数组。
    y_columns: 结果数据的字段名数组。

Returns:
    无，降维后的数据保存至Y。
'''
from pgsklearn.sql import select_from, update_set, columns_types_mapping
from sklearn.decomposition import PCA
import numpy as np

# 获取数据
records = plpy.execute(select_from(unique_key, *x_columns, table=table_name))
X = np.array([[record[column] for column in x_columns] for record in records])

# 模型训练
model = PCA(n_components = len(y_columns))
model.fit(X)
y = model.transform(X)

# 写入数据
columns_types = {record['column_name']: record['data_type'] for record in plpy.execute(columns_types_mapping(table_name))}
types = [columns_types[column] for column in y_columns] + [columns_types[unique_key]]

plan = plpy.prepare(update_set(table_name, unique_key, y_columns), types)
for record, target in zip(records, y):
  target = list(target)
  target.append(record[unique_key])
  plpy.execute(plan, target)
plpy.info(f'共处理{table_name}的{len(records)}条记录')
$$ language plpython3u;

create or replace function decomposition_isomap(table_name text, unique_key text, x_columns text[], y_columns text[]) returns void as $$
'''
Isomap降维算法

Args:
    table_name: 保存训练样本与结果的表名。
    unique_key: 字段名，需要能唯一标识样本。
    x_columns: 训练数据的字段名数组。
    y_columns: 结果数据的字段名数组。

Returns:
    无，降维后的数据保存至Y。
'''
from pgsklearn.sql import select_from, update_set, columns_types_mapping
from sklearn.manifold import Isomap
import numpy as np

# 获取数据
records = plpy.execute(select_from(unique_key, *x_columns, table=table_name))
X = np.array([[record[column] for column in x_columns] for record in records])

# 模型训练
model = Isomap(n_components = len(y_columns))
model.fit(X)
y = model.transform(X)

# 写入数据
columns_types = {record['column_name']: record['data_type'] for record in plpy.execute(columns_types_mapping(table_name))}
types = [columns_types[column] for column in y_columns] + [columns_types[unique_key]]

plan = plpy.prepare(update_set(table_name, unique_key, y_columns), types)
for record, target in zip(records, y):
  target = list(target)
  target.append(record[unique_key])
  plpy.execute(plan, target)
plpy.info(f'共处理{table_name}的{len(records)}条记录')
$$ language plpython3u;

create or replace function cluster_kmeans(table_name text, unique_key text, x_columns text[], y_column text, clusters int = 2, init text = 'k-means++', n_init int = 10, max_iter int = 300, tol float8 = 0.0001, algorithm text = 'auto') returns void as $$
from pgsklearn.sql import select_from, update_set, columns_types_mapping
from sklearn.cluster import KMeans
import numpy as np

# 获取数据
records = plpy.execute(select_from(unique_key, *x_columns, table=table_name))
X = np.array([[record[column] for column in x_columns] for record in records])

# 模型训练
model = KMeans(clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, algorithm=algorithm)
model.fit(X)
y = model.predict(X)

# 写入数据
columns_types = {record['column_name']: record['data_type'] for record in plpy.execute(columns_types_mapping(table_name))}
types = [columns_types[y_column], columns_types[unique_key]]

plan = plpy.prepare(update_set(table_name, unique_key, [y_column]), types)
for record, target in zip(records, y):
  plpy.execute(plan, [target, record[unique_key]])
plpy.info(f'共处理{table_name}的{len(records)}条记录')
$$ language plpython3u;

create or replace function cluster_gaussian_mixture(table_name text, unique_key text, x_columns text[], y_column text, clusters int = 2, covariance_type text = 'full', tol float8 = 0.001, reg_covar float8 = 0.000001, max_iter int = 100, n_init int = 1, init_params text = 'kmeans') returns void as $$
from pgsklearn.sql import select_from, update_set, columns_types_mapping
from sklearn.mixture import GaussianMixture
import numpy as np

# 获取数据
records = plpy.execute(select_from(unique_key, *x_columns, table=table_name))
X = np.array([[record[column] for column in x_columns] for record in records])

# 模型训练
model = GaussianMixture(n_components=clusters, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params)
model.fit(X)
y = model.predict(X)

# 写入数据
columns_types = {record['column_name']: record['data_type'] for record in plpy.execute(columns_types_mapping(table_name))}
types = [columns_types[y_column], columns_types[unique_key]]

plan = plpy.prepare(update_set(table_name, unique_key, [y_column]), types)
for record, target in zip(records, y):
  plpy.execute(plan, [target, record[unique_key]])
plpy.info(f'共处理{table_name}的{len(records)}条记录')
$$ language plpython3u;

create or replace function classification_gaussian_naive_bayes(table_name text, unique_key text, x_columns text[], y_column text, predict_column text, var_smoothing float = 1e-9) returns void as $$
from pgsklearn.sql import select_from, update_set, columns_types_mapping
from sklearn.naive_bayes import GaussianNB
import numpy as np

# 获取数据
records = plpy.execute(select_from(unique_key, y_column, *x_columns, table=table_name))
X = np.array([[record[column] for column in x_columns] for record in records])
y = np.array([record[y_column] for record in records])

# 模型训练
model = GaussianNB(var_smoothing=var_smoothing)
model.fit(X, y)
fit = model.predict(X)

# 写入数据
columns_types = {record['column_name']: record['data_type'] for record in plpy.execute(columns_types_mapping(table_name))}
types = [columns_types[predict_column], columns_types[unique_key]]

plan = plpy.prepare(update_set(table_name, unique_key, [predict_column]), types)
for record, target in zip(records, fit):
  plpy.execute(plan, [target, record[unique_key]])
plpy.info(f'共处理{table_name}的{len(records)}条记录')
$$ language plpython3u;

create or replace function regression_linear(table_name text, unique_key text, x_columns text[], y_column text, predict_column text, degree int = 1, fit_intercept boolean = true) returns void as $$
from pgsklearn.sql import select_from, update_set, columns_types_mapping
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

# 获取数据
records = plpy.execute(select_from(unique_key, y_column, *x_columns, table=table_name))
X = np.array([[record[column] for column in x_columns] for record in records])
y = np.array([record[y_column] for record in records])

# 模型训练
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=fit_intercept))
model.fit(X, y)
fit = model.predict(X)

# 写入数据
columns_types = {record['column_name']: record['data_type'] for record in plpy.execute(columns_types_mapping(table_name))}
types = [columns_types[predict_column], columns_types[unique_key]]

plan = plpy.prepare(update_set(table_name, unique_key, [predict_column]), types)
for record, target in zip(records, fit):
  plpy.execute(plan, [target, record[unique_key]])
plpy.info(f'共处理{table_name}的{len(records)}条记录')
$$ language plpython3u;
