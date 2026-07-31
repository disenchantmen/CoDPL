# cython: language_level=2

"""
    Copyright (c) 2013, Triad National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Triad National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from cpython.version cimport PY_MAJOR_VERSION
from libc.stdlib cimport calloc, free, malloc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import numpy as np
cimport numpy as np

# 兼容Python3的Unicode处理
cdef extern from "Python.h":
    char * PyUnicode_AsUTF8(object unicode)
    int PyUnicode_Check(object obj)

# 索引常量定义
cdef Py_ssize_t TWO_AGO = 0
cdef Py_ssize_t ONE_AGO = 1
cdef Py_ssize_t THIS_ROW = 2

cdef object _to_unicode(object s):
    """
    兼容Python3的字符串转Unicode（Python3中str就是unicode）
    """
    if PyUnicode_Check(s):
        return s
    elif isinstance(s, bytes):
        return s.decode('UTF-8', errors='replace')
    elif isinstance(s, np.str_):
        return str(s)
    elif isinstance(s, (int, float)):
        return str(s)
    raise TypeError(f'string [{s}] has an unrecognized type of [{type(s)}]')

cpdef unsigned long damerau_levenshtein_distance(seq1, seq2):
    """
    计算Damerau-Levenshtein距离（最优字符串对齐算法）
    """
    # 处理列表/元组输入
    if isinstance(seq1, (list, tuple)) and isinstance(seq2, (list, tuple)):
        s1 = seq1
        s2 = seq2
    else:
        s1 = _to_unicode(seq1)
        s2 = _to_unicode(seq2)

    # 跳过开头相同的字符
    cdef Py_ssize_t first_differing_index = 0
    cdef Py_ssize_t len_s1 = len(s1)
    cdef Py_ssize_t len_s2 = len(s2)

    while (first_differing_index < len_s1 and
           first_differing_index < len_s2 and
           s1[first_differing_index] == s2[first_differing_index]):
        first_differing_index += 1

    s1 = s1[first_differing_index:]
    s2 = s2[first_differing_index:]
    len_s1 = len(s1)
    len_s2 = len(s2)

    # 短路处理空字符串
    if len_s1 == 0:
        return len_s2
    if len_s2 == 0:
        return len_s1

    # 变量声明
    cdef Py_ssize_t i, j
    cdef Py_ssize_t offset = len_s2 + 1
    cdef unsigned long delete_cost, add_cost, substitute_cost
    cdef unsigned long * storage = NULL

    # 内存分配（3行 × (len_s2+1)列）
    storage = <unsigned long *> calloc(3 * offset, sizeof(unsigned long))
    if not storage:
        raise MemoryError("Failed to allocate memory for distance matrix")

    try:
        # 初始化当前行
        for i in range(1, offset):
            storage[THIS_ROW * offset + (i - 1)] = i

        # 迭代计算距离
        for i in range(len_s1):
            # 交换历史行数据
            for j in range(offset):
                storage[TWO_AGO * offset + j] = storage[ONE_AGO * offset + j]
                storage[ONE_AGO * offset + j] = storage[THIS_ROW * offset + j]

            # 重置当前行
            for j in range(len_s2):
                storage[THIS_ROW * offset + j] = 0
            storage[THIS_ROW * offset + len_s2] = i + 1

            # 计算编辑成本
            for j in range(len_s2):
                # 删除成本（从上到下）
                delete_cost = storage[ONE_AGO * offset + j] + 1
                # 插入成本（从左到右）
                add_cost = storage[THIS_ROW * offset + (j - 1)] + 1 if j > 0 else (i + 1) + 1
                # 替换成本
                substitute_cost = storage[ONE_AGO * offset + (j - 1)] if j > 0 else (i + 1)
                if s1[i] != s2[j]:
                    substitute_cost += 1

                # 取最小成本
                storage[THIS_ROW * offset + j] = min(delete_cost, add_cost, substitute_cost)

                # 处理换位（transposition）
                if (i > 0 and j > 0 and
                        s1[i] == s2[j - 1] and s1[i - 1] == s2[j] and
                        s1[i] != s2[j]):
                    trans_cost = storage[TWO_AGO * offset + (j - 2)] + 1 if j > 1 else 1
                    if trans_cost < storage[THIS_ROW * offset + j]:
                        storage[THIS_ROW * offset + j] = trans_cost

        # 返回最终距离
        return storage[THIS_ROW * offset + (len_s2 - 1)]

    finally:
        # 确保内存释放
        if storage:
            free(storage)

cpdef float normalized_damerau_levenshtein_distance(seq1, seq2):
    """
    归一化Damerau-Levenshtein距离（0.0-1.0）
    """
    if isinstance(seq1, (list, tuple)) and isinstance(seq2, (list, tuple)):
        max_len = max(len(seq1), len(seq2))
    else:
        s1 = _to_unicode(seq1)
        s2 = _to_unicode(seq2)
        max_len = max(len(s1), len(s2))

    # 避免除零
    if max_len == 0:
        return 0.0

    distance = damerau_levenshtein_distance(seq1, seq2)
    return float(distance) / max_len

cpdef np.ndarray[np.uint32_t, ndim=1] damerau_levenshtein_distance_ndarray(seq, np.ndarray array):
    """
    批量计算与数组中每个元素的DL距离
    """
    # 预分配结果数组
    cdef np.ndarray[np.uint32_t, ndim=1] result = np.empty(len(array), dtype=np.uint32)
    cdef Py_ssize_t i
    cdef object seq_unicode = _to_unicode(seq)

    for i in range(len(array)):
        result[i] = damerau_levenshtein_distance(seq_unicode, array[i])

    return result

cdef char ** to_cstring_array(list list_str):
    """
    将Python字符串列表转为C字符串数组（注意内存释放）
    """
    cdef Py_ssize_t len_list = len(list_str)
    cdef char ** ret = <char **> PyMem_Malloc(len_list * sizeof(char *))

    if not ret:
        raise MemoryError("Failed to allocate cstring array")

    try:
        for i in range(len_list):
            ret[i] = PyUnicode_AsUTF8(_to_unicode(list_str[i]))
        return ret
    except:
        PyMem_Free(ret)
        raise

cpdef float damerau_levenshtein_diversity(np.ndarray array):
    """
    计算数组中所有序列的平均两两DL距离
    """
    cdef unsigned long total_distance = 0
    cdef Py_ssize_t len_array = len(array)
    cdef Py_ssize_t i, j

    if len_array <= 1:
        return 0.0

    # 计算所有两两组合的距离和
    for i in range(len_array):
        for j in range(i + 1, len_array):
            total_distance += damerau_levenshtein_distance(array[i], array[j])

    # 计算平均距离（组合数：n*(n-1)/2）
    cdef float pair_count = len_array * (len_array - 1) / 2.0
    return float(total_distance) / pair_count

cpdef np.ndarray[np.float32_t, ndim=1] normalized_damerau_levenshtein_distance_ndarray(seq, np.ndarray array):
    """
    批量计算归一化DL距离
    """
    cdef np.ndarray[np.float32_t, ndim=1] result = np.empty(len(array), dtype=np.float32)
    cdef Py_ssize_t i
    cdef object seq_unicode = _to_unicode(seq)

    for i in range(len(array)):
        result[i] = normalized_damerau_levenshtein_distance(seq_unicode, array[i])

    return result