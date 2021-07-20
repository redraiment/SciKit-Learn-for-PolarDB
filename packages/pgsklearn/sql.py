def coercion(column_definitions, records):
    '''根据字段类型定义，将字符串形式的数据转换成目标数据类型'''
    def cast(type, value):
        if 'float' in type:
            return float(value)
        if 'int' in type:
            return int(value)
        return value
    types = [definition['type'] for definition in column_definitions]
    return [[cast(type, value) for type, value in zip(types, record)] for record in records]

def escape(context):
    '''字符串转义：将单引号转换成两个连续的单引号。'''
    return context.replace("'", "''")

def create_table(name, description, column_definitions):
    '''
    生成创建表格的SQL，默认包含一列自增的id。

    Args:
        name: 表名。
        description: 表描述。
        column_definitions: dict类型的字段定义。包含以下三项：
            * name: 字段名。字符串类型。必选。
            * type: 字段类型。字符串类型。必选。
            * default_value: 默认值。值类型。可选，默认为空。
            * is_nullable: 是否可为空。布尔类型。可选，默认为True。
            * comment: 字段描述。字符串类型。可选，默认为空。

    Returns:
        字符串形式的创建表格的SQL语句。
    '''

    # 可选值处理
    template = '''drop table if exists %s;
create table %s (
  id bigserial primary key%s
);
comment on table %s is '%s';
%s'''
    description = escape(description)

    columns, comments = [], []
    for definition in column_definitions:
        # 表名
        definition['table_name'] = name
        # 空
        is_nullable = definition.get('is_nullable', True)
        definition['is_nullable'] = 'not null' if is_nullable == False else 'null'
        # 默认值
        default_value = definition.get('default_value', None)
        definition['default_value'] = f'default {"null" if default_value is None else default_value}'
        # 注释
        comment = definition.get('comment', '')
        definition['comment'] = escape(comment)
        # 拼装SQL
        columns.append(',\n  %(name)s %(type)s %(is_nullable)s %(default_value)s' % definition)
        comments.append("comment on column %(table_name)s.%(name)s is '%(comment)s';\n" % definition)
    return template % (name, name, ''.join(columns), name, description, ''.join(comments))

def insert_into(table_name, column_definitions):
    '''
    生成插入记录的SQL。

    Args:
        name: 表名。
        column_definitions: dict类型的字段定义。包含以下三项：
            * name: 字段名。字符串类型。必选。
            * type: 字段类型。字符串类型。必选。

    Returns:
        (字符串形式的插入记录的SQL语句, 参数类型列表)。
    '''
    template = 'insert into %s (%s) values (%s)'
    columns = ', '.join([definition['name'] for definition in column_definitions])
    placeholders = ', '.join(f'${index}' for index in range(1, len(column_definitions) + 1))
    return template % (table_name, columns, placeholders)

def select_from(*columns, **options):
    '''
    生成Select查询SQL语句。

    Args:
        columns: 字段名列表。
        options: 其他选项。
            * table: 目标表名。

    Returns:
        字符串形式的Select查询语句。
    '''
    return f"select {', '.join(columns)} from {options['table']}"

def update_set(table, unique_key, columns):
    '''
    生成Update更新SQL语句。
    
    Args:
        table: 目标表名。
        unique_key: 唯一标识字段名。
        columns: 字段名列表。

    Returns:
        字符串形式的Update更新语句。
    '''
    size = len(columns) + 1
    columns = ', '.join([f'{name} = ${index}' for name, index in zip(columns, range(1, size))])
    return f"update {table} set {columns} where {unique_key} = ${size}"

def columns_types_mapping(table_name):
    '''
    生成查询目标表字段名及字段类型的SQL语句。

    Args:
        table_name: 目标表名。

    Returns:
        字符串形式的查询语句。
    '''
    return "select column_name, data_type from information_schema.columns where table_name = '%s'" % escape(table_name)
