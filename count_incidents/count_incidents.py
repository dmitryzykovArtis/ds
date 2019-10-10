"""
Модуль предоставляющий функцию для подсчета инцидентов согласно ТЗ.

    count_incidents
        основная функция производящая подсчет инцидентов по заданным аргументам

    generate_incident
        функция для генерации случайных данных, необходимых для тестирования count_incidents

    log_time
        вспомогательная функция для подсчета и логирования времени выполнения

    str2bool
        вспомогательная функция для перекодировки аргумента с коммандной строки в bool

    str2func
        вспомогательная функция для перекодировки аргумента с коммандной строки в один из вариантов исполняемых функций

    test
        функция для тестирования count_incidents

    verb_print
        вспомогательная функция печати скрывающая проверку условия вывода verbose

Заметки
-------
    kwargs необходим для реализации возможности вызова метода как с консоли, так и посредством import
        в случае отсутствия параметров по умолчанию, либо запуска скрипта только через консоль данный
        код можно упростить.

    для логирования рекомендуется использовать сторонний модуль, используемый для всего проекта.

    для тестирования также рекомендуется использовать отдельный модуль относящийся ко всему проекту.

    т.к. данная функция на время задерживает процесс выполнения программы, предполагается использовать
    извне в отдельном потоке. в случае консольного выполнения отдельный поток для GUI не требуется.

    возможный вариант языка документации английский, однако так как в ТЗ язык документации не указан, а само ТЗ
    написано на русском языке, принято выбрать за язык документации русский.

    в случае наличия модуля memory_profiler можно расскоментировать строки 45, 51 для оценки затрат по памяти,
    тестирование на операционной системе Windows 10, 4.00 GHz - 154.4 Mb | 5.72 c  (n = 1 000 000 , m = 100 ) - ТЗ
                                                              - 690.9 Mb | 67.66 с (n = 10 000 000, m = 1000) - ТЗ х 10
"""

import pandas as pd
import numpy as np
import time
from array import *
import argparse
import os.path
# from memory_profiler import profile

timer = 0
start = 0


# @profile
def count_incident(input_file='incidents.csv', output_file='incidents_count.csv', m=100, dt=0.3, verbose=True, kwargs={}):
    """
    Основная исполняемая функция осуществляющая подсчет инцидентов.

    Параметры
    ---------
        input_file : str, path
            Путь к файлу с данными необходимыми к обработке (default = 'incidents.csv')
        output_file : str, path
            Путь для создания файла с обработанными данными (default = 'incidents_count.csv')
        m : int
            Количество возможных уникальных значений feature1, feature2 (default = '100')
        dt : float
            Разница по времени в области которой подсчитываются инциденты (default = 0.3)
        verbose : bool, int
            Параметр определяющий тип логирования,
            целочисленная величена определяет период логирования. (default = True)
        kwargs : dict
            Словарь передающий аргументы полученные из консоли

    Возвращаемые значения
    ---------------------
        None

    Исключения
    ----------
        AssertionError - если по пути input_file файла с входными данными не существует.

    Заметки
    -------
        Основной алгоритм заключается в предварительной сортировке данных и последующем хранении для
        каждого различного по f1, f2 типа объекта последнего объекта данного типа входящего во временной
        интервал для предыдущего объекта данного типа - last_in_dt. Хранение предыдущего объекта данного типа
        last_index_for_mask_arr в зависимости от количества различных типов.
        Хранение следующего объекта данного типа next_same_mask, и построении тем самым односвязного списка объектов
        для нахождения последнего объекта входящего во временной интервал для текущего объекта за минимальное
        время. Суммарное количество инцидентов incidents_sum для текущего объекта находится как суммарное
        количество инцидентов для предыдущего объекта данного типа + 1. Количество инцидентов в заданном временном
        интервале для текущего объекта incidents_count находится как разность между количеством всех инцидентов
        incidents_sum для объектов данного типа на текущем объекте и количеством всех инцидентов incidents_sum
        для объектов данного типа на последнем объекте данного типа входящим в интервал, найденным выше.

        array был выбран как необходимая структура данных, в связи со значительным преимуществом в скорости
            записи и извлечения данных по сравнению с DataFrame и ndarray.

        исходя из того что около 50% времени работы программы занимает ввод и вывод данных, предлагается
        рассмотреть другие способы ввода, вывода данных, другие типы хранения данных.

        при значительном увеличении времени сортировки в связи с не подходящей структурой в данных
        рекомендуется изменить метод сортировки с quicksort (n^2, nlogn, n) на mergesort (nlogn)

        в случае m << n и значительного увеличения времени поиска необходимого объекта в построенном
        в данных односвязном списке, при необходимости минимизации времени за счет потребления памяти,
        предлагается использовать m массивов и осуществлять бинарный поиск для нахождения необходимого объекта.

        при возможных m > 1000 для целесообразного использования памяти рекомендуется для хранения последнего
        объекта совпадающего по типу с текущим использовать dict, не смотря на значительую потерю по времени чтения,
        записи. Возможно условное использования вместе с arr для m < 1000. При текущем ТЗ считаю данное дополнение
        лишним, в связи с чистотой кода. При уточняющем ТЗ возможна реализация.

    """

    input_file = kwargs.get('input_file', input_file)
    output_file = kwargs.get('output_file', output_file)
    m = kwargs.get('m', m)
    dt = kwargs.get('dt', dt)
    verbose = kwargs.get('verbose', verbose)

    assert os.path.exists(input_file), "INPUT FILE NOT EXIST"

    log_time('start', verbose)

    last_index_for_mask_arr = np.full((m, m), -1)
    df = pd.read_csv(input_file, index_col='id')

    log_time('load data', verbose)

    df.sort_values(by='time', inplace=True)

    log_time('sort data', verbose)

    time_arr = array('f', df['time'].to_numpy())
    n = len(time_arr)
    feature1 = array('I', df['feature1'].to_numpy())
    feature2 = array('I', df['feature2'].to_numpy())
    last_in_dt = array('L', np.zeros(n, dtype=int))
    incidents_sum = array('L', np.zeros(n, dtype=int))
    incidents_count = array('L', np.zeros(n, dtype=int))
    next_same_mask = array('L', np.zeros(n, dtype=int))

    log_time('prepare data', verbose)

    for i in range(len(df)):
        if verbose > 1 and i != 0 and i % verbose == 0:
            print('Current step: ' + str(i))

        current_time = time_arr[i]
        border_time = current_time - dt
        i_feature1 = feature1[i]
        i_feature2 = feature2[i]

        if last_index_for_mask_arr[i_feature1][i_feature2] == -1:
            last_index_for_mask_arr[i_feature1][i_feature2] = i
            last_in_dt[i] = i
            incidents_sum[i] = 0
            incidents_count[i] = 0
            continue

        last_index_for_mask = last_index_for_mask_arr[i_feature1][i_feature2]
        next_same_mask[last_index_for_mask] = i
        j = last_in_dt[last_index_for_mask]

        while time_arr[j] < border_time:
            j = next_same_mask[j]

        last_index_for_mask_arr[i_feature1][i_feature2] = i
        last_in_dt[i] = j
        incidents_sum[i] = incidents_sum[last_index_for_mask] + 1
        incidents_count[i] = incidents_sum[i] - incidents_sum[j]

    log_time('count data', verbose)
    df['count'] = incidents_count
    df.sort_index(inplace=True)

    log_time('backsort data', verbose)

    df[['count']].to_csv(output_file, index_label='id', compression='infer')

    log_time('output data', verbose)

    log_time('total_time', verbose)


def generate_incident(output_file='incidents.csv', m=100, n=1000000, kwargs={}):
    """
    Функция случайно генерирующая данные необходимые для нагрузочного тестирования count_incident.

    Параметры
    ---------
        output_file : str, path
            Путь для создания файла с сгенерированными данными (default = 'incidents.csv')
        n : int
            Количество объектов в данных (default = '1000000')
        m : int
            Количество возможных уникальных значений feature1, feature2 (default = '100')
        kwargs : dict
            Словарь передающий аргументы полученные из консоли

    Возвращаемые значения
    ---------------------
        None
    """
    output_file = kwargs.get('output_file', output_file)
    m = kwargs.get('m', m)
    n = kwargs.get('n', n)

    df = pd.DataFrame({'feature1': np.random.randint(m, size=(n,)),
                       'feature2': np.random.randint(m, size=(n,)),
                       'time': np.random.rand(n)})

    df.to_csv(output_file, index_label='id')


def log_time(s='start', verbose=True):
    """
    Вспомогательная функция для подсчета и логирования времени выполнения.

    Параметры
    ---------
        s : str
            Выводимая строка. Значение по умолчанию задает начало отсчета (default = 'start')
        verbose : bool
            Параметр определяющий тип логирования,


    Возвращаемые значения
    ---------------------
        None
    """

    global timer
    global start
    if not verbose:
        return
    if s == 'start':
        timer = time.time()
        start = timer
        return
    if s == 'total_time':
        print('{:-^31s}'.format(''))
        print('{:<15}:{:>15.2f}'.format(s, time.time() - start))
    else:
        print('{:<15}:{:>15.2f}'.format(s, time.time() - timer))
        timer = time.time()


def str2bool(v):
    """
    Вспомогательная функция для перекодировки аргумента с коммандной строки в bool

    Параметры
    ---------
        v : str
            Аргумент консоли предполагаемо описывающий тип bool

    Возвращаемые значения
    ---------------------
        bool

    Исключения
    ----------
        ArgumentTypeError - в случае передачи аргумента не интерпретируемого как Boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.isdecimal() and int(v) > 1:
        return int(v)
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2func(f):
    """
    Вспомогательная функция для перекодировки аргумента с коммандной строки в function

    Параметры
    ---------
        а : str
            Аргумент консоли предполагаемо характеризующий одну из функций модуля

    Возвращаемые значения
    ---------------------
        function

    Исключения
    ----------
        ArgumentError - в случае передачи аргумента не интерпретируемого ни как одна из функций модуля.
    """
    if f.lower() in ('count', 'c', 'count_incident'):
        return count_incident
    elif f.lower() in ('test', 't'):
        return test
    elif f.lower() in ('generate', 'g', 'generate_incident'):
        return generate_incident
    else:
        raise argparse.ArgumentError('Function not found.')


def test(verbose=True, kwargs={}):
    """
    Функция для тестирования count_incidents

    Параметры
    ---------
        input_file : str, path
            Путь к файлу с данными необходимыми к обработке (default = 'incidents.csv')
        output_file : str, path
            Путь для создания файла с обработанными данными (default = 'incidents_count.csv')
        m : int
            Количество возможных уникальных значений feature1, feature2 (default = '100')
        dt : float
            Разница по времени в области которой подсчитываются инциденты (default = 0.3)
        verbose : bool, int
            Параметр определяющий тип логирования,
            целочисленная величена определяет период логирования. (default = True)
        kwargs : dict
            Словарь передающий аргументы полученные из консоли

    Возвращаемые значения
    ---------------------
        None

    Описание тестов
    ---------------
    Тест - 1 : Проверяет корректность ответа функции на образце из технического задания
    Тест - 2 : Проверяет корректность ответа функции в крайнем случае при dt -> 0
    Тест - 3 : Нагрузочное тестирование проверяет время работы на соответствие техническому заданию
    Тест - 4 : Проверяет корректность ответа функции при n = 20

    Заметки
    -------
    Для дальнейшей работы с модулем и последующим интегрированием в систему необходимы дополнительные тесты
    осуществляющие более полное покрытие (m = 1; n = 1; dt -> 1; m<<n; m <> n; n > 1000000)
    """

    verbose = kwargs.get('verbose', verbose)

    passed = 0
    failed = 0

    verb_print('{:-^31s}'.format("TEST-1 - DT = 0.3"), verbose)
    verb_print('{:<15}'.format("INPUT-DATA:"), verbose)
    df = pd.DataFrame({'feature1': np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0]),
                       'feature2': np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1]),
                       'time': np.array([0.206520219143,
                                         0.233725001118,
                                         0.760992754734,
                                         0.92776979943,
                                         0.569711498585,
                                         0.99224586863,
                                         0.593264390713,
                                         0.694181201747,
                                         0.823812651856,
                                         0.906011017725])
                       }, dtype='int64')
    verb_print(df, verbose)
    df.to_csv('test1_input.csv', index_label='id')

    count_incident('test1_input.csv', 'test1_output.csv', verbose=False)

    verb_print('{:<15}'.format("\nFUNCTION-OUTPUT:"), verbose)
    df = pd.read_csv('test1_output.csv', index_col='id')
    verb_print(df, verbose)

    verb_print('{:<15}'.format("\nCORRECT-OUTPUT:"), verbose)
    test1_correct_output = pd.DataFrame({'count': np.array([0, 0, 0, 1, 0, 2, 0, 1, 0, 1])}, dtype='int64')
    test1_correct_output.index.names = ['id']
    verb_print(test1_correct_output, verbose)

    if test1_correct_output.equals(df):
        verb_print('{:<15}'.format("\nTEST-1 - PASSED"), verbose)
        passed += 1
    else:
        verb_print('{:<15}'.format("\nTEST-1 - FAILED"), verbose)
        failed += 1

    verb_print('\n{:-^31s}'.format("TEST-2 - DT = 0.001"), verbose)
    verb_print('{:<15}'.format("INPUT-DATA:"), verbose)
    df = pd.DataFrame({'feature1': np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0]),
                       'feature2': np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1]),
                       'time': np.array([0.206520219143,
                                         0.233725001118,
                                         0.760992754734,
                                         0.92776979943,
                                         0.569711498585,
                                         0.99224586863,
                                         0.593264390713,
                                         0.694181201747,
                                         0.823812651856,
                                         0.906011017725])
                       }, dtype='int64')
    verb_print(df, verbose)
    df.to_csv('test2_input.csv', index_label='id')

    count_incident('test2_input.csv', 'test2_output.csv', dt=0.001, verbose=False)

    verb_print('{:<15}'.format("\nFUNCTION-OUTPUT:"), verbose)
    df = pd.read_csv('test2_output.csv', index_col='id')
    verb_print(df, verbose)

    verb_print('{:<15}'.format("\nCORRECT-OUTPUT:"), verbose)
    test1_correct_output = pd.DataFrame({'count': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}, dtype='int64')
    test1_correct_output.index.names = ['id']
    verb_print(test1_correct_output, verbose)

    if test1_correct_output.equals(df):
        verb_print('{:<15}'.format("\nTEST-2 - PASSED"), verbose)
        passed += 1
    else:
        verb_print('{:<15}'.format("\nTEST-2 - FAILED"), verbose)
        failed += 1

    verb_print('\n{:-^31s}'.format("TEST-3 - TIMETEST"), verbose)
    verb_print('{:<15}'.format("GENERATE-DATA: N = 1 000 000 | M = 100 | DT = 0.3"), verbose)
    generate_incident('test3_input.csv', n=1000000, m=100)

    verb_print('{:<15}'.format("\nSTART-FUNCTION:\n"), verbose)

    start_test = time.time()
    count_incident('test3_input.csv', 'test3_output.csv', dt=0.3, verbose=verbose)
    end_test = time.time()

    if end_test - start_test < 60:
        verb_print('{:<15}'.format("\nTEST-3 - PASSED"), verbose)
        passed += 1
    else:
        verb_print('{:<15}'.format("\nTEST-3 - FAILED"), verbose)
        failed += 1

    verb_print('\n{:-^31s}'.format("TEST-4 - DT = 0.3"), verbose)
    verb_print('{:<15}'.format("INPUT-DATA:"), verbose)
    df = pd.DataFrame({'feature1': np.array([2, 2, 2, 0, 2, 0, 1, 1, 1, 1, 2, 0, 0, 2, 0, 1, 0, 1, 2, 1]),
                       'feature2': np.array([2, 0, 2, 0, 0, 2, 1, 2, 2, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0]),
                       'time': np.array([0.44584573259023697,
                                         0.42196958745536806,
                                         0.868653006167327,
                                         0.598285399140271,
                                         0.864467535617926,
                                         0.8340707640923651,
                                         0.24661704864714595,
                                         0.03940120690616733,
                                         0.6055718410545884,
                                         0.8149582914950358,
                                         0.7744040973632863,
                                         0.7156764597406549,
                                         0.1816363544687034,
                                         0.8743115073629673,
                                         0.528517161833181,
                                         0.8374942582210612,
                                         0.6538487793474304,
                                         0.5728125952008161,
                                         0.6515069203447164,
                                         0.11161506770259633
                                         ])
                       }, dtype='int64')
    verb_print(df, verbose)
    df.to_csv('test4_input.csv', index_label='id')

    count_incident('test4_input.csv', 'test4_output.csv', dt=0.3, verbose=False)

    verb_print('{:<15}'.format("\nFUNCTION-OUTPUT:"), verbose)
    df = pd.read_csv('test4_output.csv', index_col='id')
    verb_print(df, verbose)

    verb_print('{:<15}'.format("\nCORRECT-OUTPUT:"), verbose)
    test1_correct_output = pd.DataFrame({'count': np.array([0, 0, 0, 0, 2, 1, 0, 0, 0, 1,
                                                            1, 1, 0, 3, 0, 2, 0, 0, 1, 0])}, dtype='int64')
    test1_correct_output.index.names = ['id']
    verb_print(test1_correct_output, verbose)

    if test1_correct_output.equals(df):
        verb_print('{:<15}'.format("\nTEST-4 - PASSED"), verbose)
        passed += 1
    else:
        verb_print('{:<15}'.format("\nTEST-4 - FAILED"), verbose)
        failed += 1

    verb_print('{:<15}'.format("\nTOTAL PASSED: " + str(passed)), True)
    verb_print('{:<15}'.format("\nTOTAL FAILED: " + str(failed)), True)


def verb_print(s, v=True):
    """
    Вспомогательная функция печати скрывающая проверку условия вывода verbose

    Параметры
    ---------
        s : str
            Выводимая строка
        v : bool
            Условие вывода

    Возвращаемые значения
    ---------------------
    None
    """

    if not v:
        return
    print(s)


if __name__ == '__main__':
    """
    Инициализация и задание параметров парсера аргументов консоли
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input_file',
        type=str,
        help='str : Input file incidents.csv - default'
    )

    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        help='str : Output file.  incidents_count.csv - default'
    )

    parser.add_argument(
        '-v',
        '--verbose',
        type=str2bool,
        help='bool, int : Activate verbose.  True - default'
    )

    parser.add_argument(
        '-n',
        '--n',
        type=int,
        help='int : Number of rows. 1 000 000 - default'
    )

    parser.add_argument(
        '-m',
        '--m',
        type=int,
        help='int : Number of unique values in feature1 and feature2. 100 - default'
    )

    parser.add_argument(
        '-dt',
        '--dt',
        type=float,
        help='float : [0:1) Area of time before incident. 0.3 - default'
    )

    parser.add_argument(
        '-f',
        '--foo',
        type=str2func,
        default=count_incident,
        help='str : (c - count (default), t - test, g - generate) Choose function'
    )

    nargs = vars(parser.parse_args())
    foo = nargs['foo']
    # args = {k: v for k, v in nargs if v is not None and k != 'foo'} ??? don't work ??? but next is ok
    args = {k: nargs[k] for k in nargs if nargs[k] is not None and k != 'foo'}
    foo(kwargs=args)
