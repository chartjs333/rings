# Запуск Java RINGS в Windows

Нужны Git, JDK 21 или новее (команды `java` и `javac` в PATH) и Python 3.10+.
Команды ниже выполняются в PowerShell. Для самого расчёта Java не нужны C++,
CUDA или видеокарта.

## Получить и проверить код

```powershell
git clone --branch codex/java-flywire-benchmark --single-branch https://github.com/chartjs333/rings.git rings-java
cd rings-java
New-Item -ItemType Directory -Force build/java | Out-Null
javac --release 21 -d build/java java/RingsBenchmark.java
java -cp build/java RingsBenchmark --self-test
```

Успешный тест печатает JSON с `"passed":true`.

## Подготовить полную таблицу FlyWire v783

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy pyarrow pandas
.\.venv\Scripts\python.exe -c "from pathlib import Path; from scripts.download_flywire import download; download(Path('.runtime/shiu_v783'))"
.\.venv\Scripts\python.exe scripts/prepare_flywire_benchmark.py --completeness .runtime/shiu_v783/Completeness_783.csv --connectivity .runtime/shiu_v783/Connectivity_783.parquet --output .runtime/java_graph/graph.bin
```

Загрузчик проверяет размер и контрольные суммы закреплённых исходных файлов.
Итог: 138 639 узлов и 15 091 983 направленных связи. Большой бинарный граф
создаётся локально и не хранится в git.

## Измерить Java

```powershell
java -Xms1g -Xmx1g -XX:-UseFMA -cp build/java RingsBenchmark --graph .runtime/java_graph/graph.bin --mode platform --threads 4 --warmup 600 --steps 400 --stimulus 720575940624963786 --output .runtime/java_state.bin
```

Поле `steps_per_second` — число шагов всей сети в секунду. Загрузка графа,
прогрев и сохранение состояния не включены в это время. Шаг модели равен
1 мс; для реального времени нужно 1000 шагов/с. Используется `double`.

Для контроля с четырьмя виртуальными рабочими потоками:

```powershell
java -Xms1g -Xmx1g -XX:-UseFMA -Djdk.virtualThreadScheduler.parallelism=4 -Djdk.virtualThreadScheduler.maxPoolSize=4 -cp build/java RingsBenchmark --graph .runtime/java_graph/graph.bin --mode virtual --threads 4 --warmup 600 --steps 400 --stimulus 720575940624963786
```

Ограничение Java-загрузчика: бинарный файл графа должен быть меньше 2 GiB;
этот полный граф занимает около 183 MB и укладывается в ограничение.

Для сравнения Java и C++ с одинаковым выделением процессоров используйте
[общий протокол](../java/README.md) в Linux/WSL2 либо workflow GitHub Actions.
Замер на другом компьютере нельзя непосредственно делить на опубликованное
время C++: обе версии нужно запускать на одной машине.
