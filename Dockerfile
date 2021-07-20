FROM redraiment/polardb-for-postgresql:latest
MAINTAINER redraiment@gmail.com

RUN cd /home/postgres \
 && wget https://ftp.postgresql.org/pub/source/v11.2/postgresql-11.2.tar.gz \
 && tar xvf postgresql-11.2.tar.gz \
 && cd postgresql-11.2 \
 && ./configure --prefix=/usr/local \
 && make -C src/bin install \
 && make -C src/include install \
 && make -C src/interfaces install \
 && echo -e "export PGHOST=localhost\nexport PGPORT=10001\nexport PGUSER=postgres" >> /root/.bashrc

RUN cd /home/postgres \
 && git clone https://github.com/saitoha/libsixel.git \
 && cd libsixel \
 && ./configure --disable-python \
 && make install \
 && cd python \
 && python3 setup.py install \
 && pip3 install scikit-learn matplotlib seaborn

COPY packages/pgsklearn /usr/lib/python3.6/site-packages/pgsklearn
COPY extensions/* /home/postgres/polardb/polardbhome/share/extension/
COPY datasets /home/postgres/datasets
COPY images /home/postgres/images
RUN cd /home/postgres \
 && chown -R postgres:postgres datasets images \
 && chown -R postgres:postgres /home/postgres/polardb/polardbhome/share/extension/*
