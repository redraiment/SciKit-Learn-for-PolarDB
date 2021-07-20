Scikit-Learn-for-PostgreSQL
====

![PolarDB Logo](https://raw.githubusercontent.com/redraiment/SciKit-Learn-for-PostgreSQL/main/images/PolarDB-logo.png)

Scikit-Learn-for-PostgreSQL是一款基于PolarDB for PostgreSQL的数据分析工具，封装了[Scikit Learn](http://scikit-learn.org/stable/index.html)与[matplotlib](https://matplotlib.org/stable/index.html)的部分函数。目前项目用于PolarDB for PostgreSQL的教学演示，尚未在生产环境中验证，可用于学习用途，请谨慎在生产环境使用。

当前版本：1.0.0

# 选择终端

为了在psql中可视化地展示数据，需要使用可展示sixel格式图片的终端软件：

* MacOS环境：推荐使用iTerm2。
* Linux环境：推荐使用xTerm。
* Windows环境：推荐使用MinTTY。

# 安装方法

## 下载并启动Docker容器

基于PolarDB的环境已经提前构建好Docker镜像，可直接通过docker运行：

```sh
docker image pull redraiment/scikit-learn-for-polardb
docker container run -d redraiment/scikit-learn-for-polardb
sleep 60 # 等待PolarDB启动，约1分钟
docker container exec -it <container-id> psql -hlocalhost -p10001 -Upostgres
```

其中：

* 将`<container-id>`替换为你自己本地的容器id。
* 若运行psql看到错误提示：`psql: FATAL:  the database system is starting up`，请继续耐心等待。

## 安装插件

成功启动psql之后，执行以下SQL，安装插件：

```sql
create extension plpython3u;
create extension pgsklearn;
select show_logo();
```

若在终端上能看到PolarDB的logo图片，说明已经安装完成并可正常使用。如下图所示：

![show_logo](https://raw.githubusercontent.com/redraiment/SciKit-Learn-for-PostgreSQL/main/images/snapshots-show-logo.png)

# 使用示例

以下代码演示如何对鸢尾花数据集做降维，并可视化展示：

```sh
select load_iris(); -- 加载鸢尾花数据集到iris表
alter table iris add column feature1 float8, add column feature2 float8;
select decomposition_pca('iris', 'id', '{sepal_length,sepal_width,petal_length,petal_width}'::text[], '{feature1,feature2}'::text[]); -- 将4个维度的数据降维成2个维度的数据

```

若在终端上能看到如下散点图，说明降维并展示成功！

![show_logo](https://raw.githubusercontent.com/redraiment/SciKit-Learn-for-PostgreSQL/main/images/snapshots-show-lmplot.png)
