# -*- coding: utf-8 -*-
"""
mdict_indexer.py
基于原版 readmdict.py 的扩展模块，提供索引构建和快速查询功能。
不修改原版代码，通过继承实现。
"""

from __future__ import print_function, absolute_import
import sys
import os
import json
import sqlite3
from struct import pack, unpack
import zlib

# 导入原版 readmdict 中的类
try:
    from dict_vocab.readmdict import MDX, MDD
except ImportError:
    # 如果不是作为包运行，直接导入
    from dict_vocab.readmdict import MDX, MDD

# LZO 压缩支持（可选）
try:
    import lzo
except ImportError:
    lzo = None
    print("警告: LZO 模块未安装，无法处理 LZO 压缩的块")

# Python 2/3 兼容
if sys.hexversion >= 0x03000000:
    unicode = str


class ExtendedMDX(MDX):
    """
    扩展 MDX 类，添加 get_index 方法以生成索引列表和元数据。
    """
    def get_index(self, check_block=True):
        """
        遍历记录块，生成每个词条的索引信息。

        参数:
            check_block: 如果为 True，则调用父类的 _decode_block 解压并验证数据，
                         确保索引准确性（会消耗一些时间）；如果为 False，则仅读取
                         块头信息，不实际解压（速度快，但无法校验数据完整性）。

        返回:
            字典，包含以下键：
                'index_dict_list': 列表，每个元素为包含以下字段的字典：
                    - key_text: 词条文本（字符串）
                    - file_pos: 记录块在文件中的起始位置
                    - compressed_size: 压缩块大小
                    - decompressed_size: 解压后大小
                    - record_block_type: 压缩类型（0=无压缩，1=LZO，2=zlib）
                    - record_start: 本条记录在解压块中的起始偏移
                    - record_end: 本条记录在解压块中的结束偏移
                    - offset: 当前记录块之前所有记录块的总解压大小（用于计算记录在块内的偏移）
                'meta': 元数据字典，包含编码、样式表、标题、描述等。
        """
        f = open(self._fname, 'rb')
        f.seek(self._record_block_offset)

        # 读取记录块头部信息（不同版本格式略有差异）
        if self._version >= 3.0:
            # MDict 3.0 格式：块数用 4 字节整数，后面紧跟总大小（忽略）
            num_record_blocks = unpack('>I', f.read(4))[0]
            total_size = self._read_number(f)   # 总大小，此处不关心
        else:
            num_record_blocks = self._read_number(f)
            num_entries = self._read_number(f)  # 应与 self._num_entries 一致
            record_block_info_size = self._read_number(f)
            record_block_size = self._read_number(f)

        # 读取每个记录块的信息（压缩前/后大小）
        record_block_info_list = []
        for _ in range(num_record_blocks):
            if self._version >= 3.0:
                compressed_size = unpack('>I', f.read(4))[0]
                decompressed_size = unpack('>I', f.read(4))[0]
            else:
                compressed_size = self._read_number(f)
                decompressed_size = self._read_number(f)
            record_block_info_list.append((compressed_size, decompressed_size))

        index_dict_list = []
        offset = 0          # 所有已处理记录块的总解压大小
        key_idx = 0         # 当前处理的词条在 _key_list 中的索引

        for compressed_size, decompressed_size in record_block_info_list:
            current_pos = f.tell()                 # 记录块的起始文件位置
            block_compressed = f.read(compressed_size)

            # 获取压缩类型（前4字节）
            record_block_type = block_compressed[:4]
            if record_block_type == b'\x00\x00\x00\x00':
                blk_type = 0
            elif record_block_type == b'\x01\x00\x00\x00':
                blk_type = 1
            elif record_block_type == b'\x02\x00\x00\x00':
                blk_type = 2
            else:
                raise Exception('未知的压缩类型: %r' % record_block_type)

            # 如果 check_block 为 True，则调用父类 _decode_block 解压并验证
            if check_block:
                # _decode_block 会处理解密和解压，并返回完整数据
                block_decompressed = self._decode_block(block_compressed, decompressed_size)
                # 验证解压后长度
                assert len(block_decompressed) == decompressed_size

            # 根据 _key_list 切分当前块中的记录
            while key_idx < len(self._key_list):
                record_start, key_text = self._key_list[key_idx]
                # 如果记录的起始偏移已经超出当前块，则跳出处理下一个块
                if record_start - offset >= decompressed_size:
                    break

                # 计算记录结束位置
                if key_idx + 1 < len(self._key_list):
                    record_end = self._key_list[key_idx + 1][0]
                else:
                    record_end = decompressed_size + offset

                # 构建索引条目
                index_dict = {
                    'file_pos': current_pos,
                    'compressed_size': compressed_size,
                    'decompressed_size': decompressed_size,
                    'record_block_type': blk_type,
                    'record_start': record_start,
                    'record_end': record_end,
                    'offset': offset,
                    'key_text': key_text.decode('utf-8')   # key_text 已是 UTF-8 字节串
                }
                index_dict_list.append(index_dict)
                key_idx += 1

            offset += decompressed_size

        f.close()

        # 收集元数据
        # 标题和描述可能以字节形式存在于 header 中
        title = self.header.get(b'Title', b'').decode('utf-8', errors='ignore')
        description = self.header.get(b'Description', b'').decode('utf-8', errors='ignore')
        meta = {
            'encoding': self._encoding,
            'stylesheet': json.dumps(self._stylesheet, ensure_ascii=False),
            'title': title,
            'description': description,
            'version': '1.0'   # 索引器版本
        }
        return {'index_dict_list': index_dict_list, 'meta': meta}


class ExtendedMDD(MDD):
    """
    扩展 MDD 类，添加 get_index 方法。
    """
    def get_index(self, check_block=True):
        """
        与 ExtendedMDX.get_index 类似，但返回的元数据中 stylesheet 为空。
        """
        f = open(self._fname, 'rb')
        f.seek(self._record_block_offset)

        if self._version >= 3.0:
            num_record_blocks = unpack('>I', f.read(4))[0]
            total_size = self._read_number(f)
        else:
            num_record_blocks = self._read_number(f)
            num_entries = self._read_number(f)
            record_block_info_size = self._read_number(f)
            record_block_size = self._read_number(f)

        record_block_info_list = []
        for _ in range(num_record_blocks):
            if self._version >= 3.0:
                compressed_size = unpack('>I', f.read(4))[0]
                decompressed_size = unpack('>I', f.read(4))[0]
            else:
                compressed_size = self._read_number(f)
                decompressed_size = self._read_number(f)
            record_block_info_list.append((compressed_size, decompressed_size))

        index_dict_list = []
        offset = 0
        key_idx = 0

        for compressed_size, decompressed_size in record_block_info_list:
            current_pos = f.tell()
            block_compressed = f.read(compressed_size)

            record_block_type = block_compressed[:4]
            if record_block_type == b'\x00\x00\x00\x00':
                blk_type = 0
            elif record_block_type == b'\x01\x00\x00\x00':
                blk_type = 1
            elif record_block_type == b'\x02\x00\x00\x00':
                blk_type = 2
            else:
                raise Exception('未知的压缩类型: %r' % record_block_type)

            if check_block:
                block_decompressed = self._decode_block(block_compressed, decompressed_size)
                assert len(block_decompressed) == decompressed_size

            while key_idx < len(self._key_list):
                record_start, key_text = self._key_list[key_idx]
                if record_start - offset >= decompressed_size:
                    break
                if key_idx + 1 < len(self._key_list):
                    record_end = self._key_list[key_idx + 1][0]
                else:
                    record_end = decompressed_size + offset

                index_dict = {
                    'file_pos': current_pos,
                    'compressed_size': compressed_size,
                    'decompressed_size': decompressed_size,
                    'record_block_type': blk_type,
                    'record_start': record_start,
                    'record_end': record_end,
                    'offset': offset,
                    'key_text': key_text.decode('utf-8')
                }
                index_dict_list.append(index_dict)
                key_idx += 1

            offset += decompressed_size

        f.close()

        title = self.header.get(b'Title', b'').decode('utf-8', errors='ignore')
        description = self.header.get(b'Description', b'').decode('utf-8', errors='ignore')
        meta = {
            'encoding': self._encoding,
            'stylesheet': '{}',
            'title': title,
            'description': description,
            'version': '1.0'
        }
        return {'index_dict_list': index_dict_list, 'meta': meta}


class IndexBuilder(object):
    """
    索引构建器：基于扩展后的 MDX/MDD 类，构建 SQLite 索引数据库并提供查询接口。
    """
    def __init__(self, fname, encoding='', passcode=None, force_rebuild=False,
                 sql_index=True, check=False):
        """
        参数:
            fname: MDX 文件路径
            encoding: 覆盖文件头中指定的编码（可选）
            passcode: 元组 (regcode, userid) 用于解密加密文件
            force_rebuild: 强制重建索引数据库（即使已存在）
            sql_index: 是否在数据库表上创建索引以加速查询
            check: 构建索引时是否校验块数据（调用 _decode_block 解压验证）
        """
        self._mdx_file = os.path.abspath(fname)
        base, ext = os.path.splitext(self._mdx_file)
        assert ext.lower() == '.mdx'
        assert os.path.isfile(self._mdx_file)

        self._passcode = passcode
        self._encoding = encoding
        self._sql_index = sql_index
        self._check = check

        # MDX 数据库路径
        self._mdx_db = base + '.mdx.db'
        # MDD 文件及数据库（如果存在）
        self._mdd_file = base + '.mdd'
        self._mdd_db = base + '.mdd.db' if os.path.exists(self._mdd_file) else None

        # 保存 ExtendedMDX 和 ExtendedMDD 实例，用于后续数据提取
        self._mdx_obj = None
        self._mdd_obj = None

        # 构建或加载索引
        self._prepare_indexes(force_rebuild)

    def _prepare_indexes(self, force_rebuild):
        """确保索引数据库存在，并加载元数据到实例变量。"""
        # 处理 MDX
        if force_rebuild or not os.path.exists(self._mdx_db):
            self._build_mdx_index()
        else:
            # 从现有数据库加载元数据
            self._load_meta_from_db(self._mdx_db)

        # 处理 MDD（如果存在）
        if self._mdd_file and os.path.exists(self._mdd_file):
            if force_rebuild or not os.path.exists(self._mdd_db):
                self._build_mdd_index()

    def _build_mdx_index(self):
        """构建 MDX 索引数据库。"""
        print("正在构建 MDX 索引: %s" % self._mdx_db)
        # 创建扩展 MDX 实例
        mdx = ExtendedMDX(self._mdx_file, encoding=self._encoding, passcode=self._passcode)
        self._mdx_obj = mdx   # 保留实例供后续查询
        result = mdx.get_index(check_block=self._check)
        index_list = result['index_dict_list']
        meta = result['meta']

        # 写入 SQLite
        conn = sqlite3.connect(self._mdx_db)
        c = conn.cursor()
        c.execute('''CREATE TABLE MDX_INDEX (
            key_text TEXT NOT NULL,
            file_pos INTEGER,
            compressed_size INTEGER,
            decompressed_size INTEGER,
            record_block_type INTEGER,
            record_start INTEGER,
            record_end INTEGER,
            offset INTEGER
        )''')
        tuples = [(item['key_text'], item['file_pos'], item['compressed_size'],
                   item['decompressed_size'], item['record_block_type'],
                   item['record_start'], item['record_end'], item['offset'])
                  for item in index_list]
        c.executemany('INSERT INTO MDX_INDEX VALUES (?,?,?,?,?,?,?,?)', tuples)

        c.execute('CREATE TABLE META (key TEXT, value TEXT)')
        c.executemany('INSERT INTO META VALUES (?,?)', [
            ('encoding', meta['encoding']),
            ('stylesheet', meta['stylesheet']),
            ('title', meta['title']),
            ('description', meta['description']),
            ('version', meta['version'])
        ])

        if self._sql_index:
            c.execute('CREATE INDEX key_index ON MDX_INDEX (key_text)')

        conn.commit()
        conn.close()

        # 更新实例变量中的元数据
        self._encoding = meta['encoding']
        self._stylesheet = json.loads(meta['stylesheet'])
        self._title = meta['title']
        self._description = meta['description']

    def _build_mdd_index(self):
        """构建 MDD 索引数据库。"""
        if os.path.exists(self._mdd_db):
            os.remove(self._mdd_db)
        
        print("正在构建 MDD 索引: %s" % self._mdd_db)
        mdd = ExtendedMDD(self._mdd_file, passcode=self._passcode)
        self._mdd_obj = mdd
        result = mdd.get_index(check_block=self._check)
        index_list = result['index_dict_list']

        conn = sqlite3.connect(self._mdd_db)
        c = conn.cursor()
        c.execute('''CREATE TABLE MDX_INDEX (
            key_text TEXT NOT NULL UNIQUE,
            file_pos INTEGER,
            compressed_size INTEGER,
            decompressed_size INTEGER,
            record_block_type INTEGER,
            record_start INTEGER,
            record_end INTEGER,
            offset INTEGER
        )''')
        tuples = [(item['key_text'], item['file_pos'], item['compressed_size'],
                   item['decompressed_size'], item['record_block_type'],
                   item['record_start'], item['record_end'], item['offset'])
                  for item in index_list]
        c.executemany('INSERT INTO MDX_INDEX VALUES (?,?,?,?,?,?,?,?)', tuples)

        if self._sql_index:
            c.execute('CREATE UNIQUE INDEX key_index ON MDX_INDEX (key_text)')

        conn.commit()
        conn.close()

    def _load_meta_from_db(self, db_path):
        """从已存在的 SQLite 数据库加载元数据到实例变量。"""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        try:
            c.execute("SELECT key, value FROM META")
            for key, value in c:
                if key == 'encoding':
                    self._encoding = value
                elif key == 'stylesheet':
                    self._stylesheet = json.loads(value)
                elif key == 'title':
                    self._title = value
                elif key == 'description':
                    self._description = value
        except sqlite3.OperationalError:
            # 可能没有 META 表（旧版数据库），忽略
            pass
        conn.close()

    # ---------- 查询接口 ----------
    def mdx_lookup(self, keyword, ignorecase=False):
        """
        查询 MDX 词条。

        参数:
            keyword: 查询关键词
            ignorecase: 是否忽略大小写（会影响性能，因为数据库索引区分大小写）

        返回:
            释义列表（字符串），每个元素对应一个匹配词条（通常只有一个，除非文件有重复词条）。
        """
        indexes = self._lookup_indexes(self._mdx_db, keyword, ignorecase)
        if not indexes:
            return []
        # 确保 _mdx_obj 已存在，否则创建临时实例
        if self._mdx_obj is None:
            self._mdx_obj = ExtendedMDX(self._mdx_file, encoding=self._encoding,
                                        passcode=self._passcode)
        results = []
        for idx in indexes:
            data = self._extract_data(self._mdx_obj, idx)
            # 解码并处理样式
            text = data.decode(self._encoding, errors='ignore').strip('\x00')
            if self._stylesheet:
                text = self._replace_stylesheet(text)
            results.append(text)
        return results

    def mdd_lookup(self, keyword, ignorecase=False):
        """
        查询 MDD 资源。

        参数:
            keyword: 资源路径（如 "\\image.png"）
            ignorecase: 是否忽略大小写

        返回:
            二进制数据列表。
        """
        if not self._mdd_db:
            return []
        indexes = self._lookup_indexes(self._mdd_db, keyword, ignorecase)
        if not indexes:
            return []
        if self._mdd_obj is None:
            self._mdd_obj = ExtendedMDD(self._mdd_file, passcode=self._passcode)
        results = []
        for idx in indexes:
            data = self._extract_data(self._mdd_obj, idx)
            results.append(data)
        return results

    def _lookup_indexes(self, db_path, keyword, ignorecase):
        """从指定数据库查询关键词对应的索引列表。"""
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        if ignorecase:
            # 使用 LOWER 函数会使索引失效，适合小数据量
            sql = 'SELECT * FROM MDX_INDEX WHERE lower(key_text) = lower(?)'
        else:
            sql = 'SELECT * FROM MDX_INDEX WHERE key_text = ?'
        c.execute(sql, (keyword,))
        rows = c.fetchall()
        conn.close()
        indexes = []
        for row in rows:
            indexes.append({
                'file_pos': row[1],
                'compressed_size': row[2],
                'decompressed_size': row[3],
                'record_block_type': row[4],
                'record_start': row[5],
                'record_end': row[6],
                'offset': row[7]
            })
        return indexes

    def _extract_data(self, mdict_obj, index):
        """
        根据索引，使用 MDict 对象的 _decode_block 方法提取数据。
        此方法复用原版的解密/解压逻辑。
        """
        with open(mdict_obj._fname, 'rb') as f:
            f.seek(index['file_pos'])
            block_compressed = f.read(index['compressed_size'])
            # 调用原版 _decode_block 获取完整解压数据
            block_decompressed = mdict_obj._decode_block(
                block_compressed, index['decompressed_size'])
        start = index['record_start'] - index['offset']
        end = index['record_end'] - index['offset']
        return block_decompressed[start:end]

    def _replace_stylesheet(self, text):
        """替换样式标记。"""
        import re
        # 样式标记形如 `1`，分割文本
        parts = re.split(r'`\d+`', text)
        tags = re.findall(r'`\d+`', text)
        styled = parts[0]
        for i, part in enumerate(parts[1:]):
            style = self._stylesheet.get(tags[i][1:-1], ('', ''))
            if part and part.endswith('\n'):
                styled += style[0] + part.rstrip() + style[1] + '\r\n'
            else:
                styled += style[0] + part + style[1]
        return styled

    def get_mdx_keys(self, pattern=''):
        """
        获取 MDX 所有键，支持通配符（* 匹配任意字符）。

        参数:
            pattern: 匹配模式，例如 'ab*' 表示以 'ab' 开头；若为空则返回所有键。

        返回:
            键列表。
        """
        return self._get_keys(self._mdx_db, pattern)

    def get_mdd_keys(self, pattern=''):
        """获取 MDD 所有键，支持通配符。"""
        if self._mdd_db:
            return self._get_keys(self._mdd_db, pattern)
        return []

    def _get_keys(self, db_path, pattern):
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        if pattern:
            sql_pattern = pattern.replace('*', '%')
            # SQL LIKE 中的 % 匹配任意字符，_ 匹配单个字符，这里只支持 *
            c.execute('SELECT key_text FROM MDX_INDEX WHERE key_text LIKE ?',
                      (sql_pattern,))
        else:
            c.execute('SELECT key_text FROM MDX_INDEX')
        keys = [row[0] for row in c]
        conn.close()
        return keys

    # 属性访问，方便获取元数据
    @property
    def title(self):
        return self._title

    @property
    def description(self):
        return self._description

    @property
    def encoding(self):
        return self._encoding


# 如果直接运行此模块，可进行简单测试
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mdict_indexer.py <mdx_file> [keyword]")
        sys.exit(1)
    mdx_file = sys.argv[1]
    builder = IndexBuilder(mdx_file, force_rebuild=False)
    print("Title:", builder.title)
    print("Description:", builder.description)
    print("Encoding:", builder.encoding)
    if len(sys.argv) >= 3:
        keyword = sys.argv[2]
        results = builder.mdx_lookup(keyword)
        for r in results:
            print(r)
    else:
        # 打印前10个键
        keys = builder.get_mdx_keys()[:10]
        print("First 10 keys:", keys)