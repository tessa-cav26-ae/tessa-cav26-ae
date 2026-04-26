# ===================================================================
# Artifact reproduction for paper figures
# ===================================================================
#
# Reproduces all data consumed by ../mct/figures/eval-*.tex.
# Results stored locally under reproduced/ (or smoke/ with SMOKE=1).
# Existing experiment data is reused; delete the output directory to force a rerun.
#
# Usage:
#   make -f reproduce.mk                       # full reproduction
#   make -f reproduce.mk herman                # single suite
#   make -f reproduce.mk SMOKE=1               # smoke test (~10 min)
#   make -f reproduce.mk SMOKE=1 herman        # smoke single suite

PYTHON ?= python
CUDA_DEVICE ?= 0
NUM_TIMED_RUNS ?= 1
NUM_WORK_RUNS ?= 3
DTYPE ?= float64
STORM_EXTRA_ARGS ?= -tm --sylvan:threads 1

VERIFIER := $(PYTHON) -m src.postprocess
RUNNER := $(PYTHON) -m src.benchmarks
TESSA_JAX_BACKEND := jax:cuda:$(CUDA_DEVICE)

ifdef SMOKE
  TO ?= 60
  NUM_WORK_RUNS ?= 1
  RESULTS_ROOT ?= smoke
else
  TO ?= 1260
  NUM_WORK_RUNS ?= 3
  RESULTS_ROOT ?= reproduced
endif

COMMON_FLAGS := --timeout $(TO) --num-work-runs $(NUM_WORK_RUNS)
TESSA_COMMON_FLAGS := --num-timed-runs $(NUM_TIMED_RUNS) --dtype $(DTYPE)
RUNNER_TESSA := $(RUNNER) --tool tessa --model-type jani --backend $(TESSA_JAX_BACKEND) $(COMMON_FLAGS) $(TESSA_COMMON_FLAGS)
RUNNER_STORM_ADD := $(RUNNER) --tool storm --engine add $(COMMON_FLAGS)$(if $(STORM_EXTRA_ARGS), --storm-extra-args '$(STORM_EXTRA_ARGS)')
RUNNER_STORM_SPM := $(RUNNER) --tool storm --engine spm $(COMMON_FLAGS)$(if $(STORM_EXTRA_ARGS), --storm-extra-args '$(STORM_EXTRA_ARGS)')

FULL_HORIZONS := 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000


# --- Parameters (full vs smoke) ---

ifdef SMOKE
  # Herman testn
  HERMAN_TESTN_N_TESSA := 3,5
  HERMAN_TESTN_N_STORM_ADD := 3,5
  HERMAN_TESTN_N_STORM_SPM := 3,5
  HERMAN_TESTN_H := 100

  # Herman testh
  HERMAN_TESTH_N := 17
  HERMAN_TESTH_H_TESSA := 10,20
  HERMAN_TESTH_H_STORM_ADD := 10,20
  HERMAN_TESTH_H_STORM_SPM := 10,20

  # Meeting testn
  MEETING_TESTN_N_TESSA := 2,3
  MEETING_TESTN_N_STORM_ADD := 2,3
  MEETING_TESTN_N_STORM_SPM := 2,3
  MEETING_TESTN_H := 10

  # Meeting testh
  MEETING_TESTH_N := 12
  MEETING_TESTH_H_TESSA := 1,10
  MEETING_TESTH_H_STORM_ADD := 1,10
  MEETING_TESTH_H_STORM_SPM := 1,10

  # Weather Factory testn
  WF_TESTN_N_TESSA := 2,7
  WF_TESTN_N_STORM_ADD := 2,7
  WF_TESTN_N_STORM_SPM := 2,7
  WF_TESTN_H := 10

  # Weather Factory testh
  WF_TESTH_N := 13
  WF_TESTH_H_TESSA := 10,20
  WF_TESTH_H_STORM_ADD := 10,20
  WF_TESTH_H_STORM_SPM := 10,20

  # Parqueues testq
  PQ_TESTQ_Q_TESSA := 3,4
  PQ_TESTQ_Q_STORM_ADD := 3,4
  PQ_TESTQ_Q_STORM_SPM := 3,4
  PQ_TESTQ_N := 3
  PQ_TESTQ_H := 10

  # Parqueues testh
  PQ_TESTH_Q := 9
  PQ_TESTH_N := 3
  PQ_TESTH_H_TESSA := 10,20
  PQ_TESTH_H_STORM_ADD := 10,20
  PQ_TESTH_H_STORM_SPM := 10,20

else

  # Herman testn (H=100)
  #   N     storm.add    storm.spm
  #   13       39.8s        2.0s
  #   15      234.4s       19.3s
  #   17     TIMEOUT      189.1s
  #   19     TIMEOUT     TIMEOUT
  HERMAN_TESTN_H := 100
  HERMAN_TESTN_N_TESSA := 3,5,7,9,11,13,15,17,19
  HERMAN_TESTN_N_STORM_ADD := 3,5,7,9,11,13,15,17,19
  HERMAN_TESTN_N_STORM_SPM := 3,5,7,9,11,13,15,17,19

  # Herman testh (N=17)
  #   H     storm.add    storm.spm
  #    90    1189.3s      185.2s
  #   100    TIMEOUT      193.5s
  #   200    TIMEOUT      262.0s
  #   300    TIMEOUT      325.3s
  #   1000      —         673.9s
  HERMAN_TESTH_N := 17
  HERMAN_TESTH_H_TESSA := $(FULL_HORIZONS)
  HERMAN_TESTH_H_STORM_ADD := 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300
  HERMAN_TESTH_H_STORM_SPM := $(FULL_HORIZONS)

  # Meeting testn (H=10)
  #   N     storm.add    storm.spm
  #    9       51.3s        1.6s
  #   10      291.7s        7.8s
  #   11     FAILED        39.8s
  #   12     FAILED       FAILED
  MEETING_TESTN_H := 10
  MEETING_TESTN_N_TESSA := 2,3,4,5,6,7,8,9,10,11,12
  MEETING_TESTN_N_STORM_ADD := 2,3,4,5,6,7,8,9,10,11,12
  MEETING_TESTN_N_STORM_SPM := 2,3,4,5,6,7,8,9,10,11,12

  # Meeting testh (N=12)
  #   H     storm.add    storm.spm
  #     1   FAILED       152.7s
  #     2   FAILED       174.0s
  #    10     —          181.7s
  #   100     —          218.2s
  #   1000    —          527.8s
  MEETING_TESTH_N := 12
  MEETING_TESTH_H_TESSA := $(FULL_HORIZONS)
  MEETING_TESTH_H_STORM_ADD := 1,2
  MEETING_TESTH_H_STORM_SPM := $(FULL_HORIZONS)

  # Weather Factory testn (H=10)
  #   N     storm.add    storm.spm
  #    7        2.4s        0.1s
  #    8        3.8s        0.2s
  #    9        9.8s        0.6s
  #   10       37.8s        2.1s
  #   11      188.9s        8.7s
  #   12     FAILED        35.8s
  #   13     FAILED       147.3s
  #   15       —         TIMEOUT
  #   16       —         TIMEOUT
  WF_TESTN_N_TESSA := 2,7,8,9,10,11,12,13,15,16
  WF_TESTN_N_STORM_ADD := 2,7,8,9,10,11,12,13
  WF_TESTN_N_STORM_SPM := 2,7,8,9,10,11,12,13,15,16
  WF_TESTN_H := 10

  # Weather Factory testh (N=13)
  #   H     storm.add    storm.spm
  #     1   FAILED       142.4s
  #     2   FAILED       143.5s
  #    10     —          148.2s
  #   100     —          177.7s
  #   700     —          375.8s
  #  1000     —          481.2s
  WF_TESTH_N := 13
  WF_TESTH_H_TESSA := $(FULL_HORIZONS)
  WF_TESTH_H_STORM_ADD := 1,2
  WF_TESTH_H_STORM_SPM := $(FULL_HORIZONS)

  # Parqueues testq (N=3, H=10)
  #   Q     storm.add    storm.spm
  #    8       43.5s        3.1s
  #    9      302.0s       20.5s
  #   10     TIMEOUT      FAILED
  #   11     TIMEOUT      FAILED
  PQ_TESTQ_N := 3
  PQ_TESTQ_H := 10
  PQ_TESTQ_Q_TESSA := 3,4,5,6,7,8,9,10,11
  PQ_TESTQ_Q_STORM_ADD := $(PQ_TESTQ_Q_TESSA)
  PQ_TESTQ_Q_STORM_SPM := $(PQ_TESTQ_Q_TESSA)

  # Parqueues testh (Q=9, N=3)
  #   H     storm.add    storm.spm
  #    10      299.8s       21.2s
  #    30     1120.1s       21.1s
  #    40    TIMEOUT        21.6s
  #   100    TIMEOUT        25.2s
  #   1000   TIMEOUT        70.6s
  PQ_TESTH_Q := 9
  PQ_TESTH_N := 3
  PQ_TESTH_H_TESSA := $(FULL_HORIZONS)
  PQ_TESTH_H_STORM_ADD := 1,2,3,4,5,6,7,8,9,10,20,30,40,50
  PQ_TESTH_H_STORM_SPM := $(FULL_HORIZONS)
endif

# --- Targets ---

.PHONY: all herman meeting weather-factory parqueues \
	tessa storm-add storm-spm \
	herman.testn herman.testh meeting.testn meeting.testh \
	weather-factory.testn weather-factory.testh \
	parqueues.testq parqueues.testh \
	herman.testn.tessa herman.testn.storm-add herman.testn.storm-spm \
	herman.testh.tessa herman.testh.storm-add herman.testh.storm-spm \
	meeting.testn.tessa meeting.testn.storm-add meeting.testn.storm-spm \
	meeting.testh.tessa meeting.testh.storm-add meeting.testh.storm-spm \
	weather-factory.testn.tessa weather-factory.testn.storm-add weather-factory.testn.storm-spm \
	weather-factory.testh.tessa weather-factory.testh.storm-add weather-factory.testh.storm-spm \
	parqueues.testq.tessa parqueues.testq.storm-add parqueues.testq.storm-spm \
	parqueues.testh.tessa parqueues.testh.storm-add parqueues.testh.storm-spm

all: herman meeting weather-factory parqueues

# --- Tool-aggregate targets ---

tessa: herman.testn.tessa herman.testh.tessa \
	meeting.testn.tessa meeting.testh.tessa \
	weather-factory.testn.tessa weather-factory.testh.tessa \
	parqueues.testq.tessa parqueues.testh.tessa

storm-add: herman.testn.storm-add herman.testh.storm-add \
	meeting.testn.storm-add meeting.testh.storm-add \
	weather-factory.testn.storm-add weather-factory.testh.storm-add \
	parqueues.testq.storm-add parqueues.testh.storm-add

storm-spm: herman.testn.storm-spm herman.testh.storm-spm \
	meeting.testn.storm-spm meeting.testh.storm-spm \
	weather-factory.testn.storm-spm weather-factory.testh.storm-spm \
	parqueues.testq.storm-spm parqueues.testh.storm-spm

# --- Clean targets ---

clean.tessa: clean.herman.testn.tessa clean.herman.testh.tessa \
	clean.meeting.testn.tessa clean.meeting.testh.tessa \
	clean.weather-factory.testn.tessa clean.weather-factory.testh.tessa \
	clean.parqueues.testq.tessa clean.parqueues.testh.tessa

clean.storm-add: clean.herman.testn.storm-add clean.herman.testh.storm-add \
	clean.meeting.testn.storm-add clean.meeting.testh.storm-add \
	clean.weather-factory.testn.storm-add clean.weather-factory.testh.storm-add \
	clean.parqueues.testq.storm-add clean.parqueues.testh.storm-add

clean.storm-spm: clean.herman.testn.storm-spm clean.herman.testh.storm-spm \
	clean.meeting.testn.storm-spm clean.meeting.testh.storm-spm \
	clean.weather-factory.testn.storm-spm clean.weather-factory.testh.storm-spm \
	clean.parqueues.testq.storm-spm clean.parqueues.testh.storm-spm

clean.herman.testn: clean.herman.testn.tessa clean.herman.testn.storm-add clean.herman.testn.storm-spm

clean.herman.testh: clean.herman.testh.tessa clean.herman.testh.storm-add clean.herman.testh.storm-spm

clean.herman: clean.herman.testn clean.herman.testh

clean.meeting.testn: clean.meeting.testn.tessa clean.meeting.testn.storm-add clean.meeting.testn.storm-spm

clean.meeting.testh: clean.meeting.testh.tessa clean.meeting.testh.storm-add clean.meeting.testh.storm-spm

clean.meeting: clean.meeting.testn clean.meeting.testh

clean.weather-factory.testn: clean.weather-factory.testn.tessa clean.weather-factory.testn.storm-add clean.weather-factory.testn.storm-spm

clean.weather-factory.testh: clean.weather-factory.testh.tessa clean.weather-factory.testh.storm-add clean.weather-factory.testh.storm-spm

clean.weather-factory: clean.weather-factory.testn clean.weather-factory.testh

clean.parqueues.testq: clean.parqueues.testq.tessa clean.parqueues.testq.storm-add clean.parqueues.testq.storm-spm

clean.parqueues.testh: clean.parqueues.testh.tessa clean.parqueues.testh.storm-add clean.parqueues.testh.storm-spm

clean.parqueues: clean.parqueues.testq clean.parqueues.testh

clean.verify:
	rm -f $(RESULTS_ROOT)/herman/testn/verify.csv $(RESULTS_ROOT)/herman/testn/verify.log
	rm -f $(RESULTS_ROOT)/herman/testh/verify.csv $(RESULTS_ROOT)/herman/testh/verify.log
	rm -f $(RESULTS_ROOT)/meeting/testn/verify.csv $(RESULTS_ROOT)/meeting/testn/verify.log
	rm -f $(RESULTS_ROOT)/meeting/testh/verify.csv $(RESULTS_ROOT)/meeting/testh/verify.log
	rm -f $(RESULTS_ROOT)/weather-factory/testn/verify.csv $(RESULTS_ROOT)/weather-factory/testn/verify.log
	rm -f $(RESULTS_ROOT)/weather-factory/testh/verify.csv $(RESULTS_ROOT)/weather-factory/testh/verify.log
	rm -f $(RESULTS_ROOT)/parqueues/testq/verify.csv $(RESULTS_ROOT)/parqueues/testq/verify.log
	rm -f $(RESULTS_ROOT)/parqueues/testh/verify.csv $(RESULTS_ROOT)/parqueues/testh/verify.log

# --- Herman ---

herman: herman.testn herman.testh

herman.testn: herman.testn.tessa herman.testn.storm-add herman.testn.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/herman/testn/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/herman/testn/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/herman/testn/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/herman/testn/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/herman/testn/verify.csv
	plot --csv $(RESULTS_ROOT)/herman/testn/tessa.csv \
		$(RESULTS_ROOT)/herman/testn/storm.add.csv \
		$(RESULTS_ROOT)/herman/testn/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x N --y work_seconds elapsed_seconds elapsed_seconds \
		--title "Herman — scaling with N" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/herman/testn/herman-testn.png
	plot --csv $(RESULTS_ROOT)/herman/testn/tessa.csv \
		$(RESULTS_ROOT)/herman/testn/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x N --y work_seconds measured_avg_seconds \
		--title "Herman — scaling with N (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/herman/testn/herman-testn-tessa.png

herman.testn.tessa:
	test -f $(RESULTS_ROOT)/herman/testn/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/herman/testn/tessa.log --output-dir $(RESULTS_ROOT)/herman/testn herman -N $(HERMAN_TESTN_N_TESSA) -H $(HERMAN_TESTN_H)

herman.testn.storm-add:
	test -f $(RESULTS_ROOT)/herman/testn/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/herman/testn/storm.add.log --output-dir $(RESULTS_ROOT)/herman/testn herman -N $(HERMAN_TESTN_N_STORM_ADD) -H $(HERMAN_TESTN_H)

herman.testn.storm-spm:
	test -f $(RESULTS_ROOT)/herman/testn/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/herman/testn/storm.spm.log --output-dir $(RESULTS_ROOT)/herman/testn herman -N $(HERMAN_TESTN_N_STORM_SPM) -H $(HERMAN_TESTN_H)

clean.herman.testn.tessa:
	rm -f $(RESULTS_ROOT)/herman/testn/tessa.csv $(RESULTS_ROOT)/herman/testn/tessa.log
	rm -f $(RESULTS_ROOT)/herman/testn/*tessa*.time.jsonl

clean.herman.testn.storm-add:
	rm -f $(RESULTS_ROOT)/herman/testn/storm.add.csv $(RESULTS_ROOT)/herman/testn/storm.add.log

clean.herman.testn.storm-spm:
	rm -f $(RESULTS_ROOT)/herman/testn/storm.spm.csv $(RESULTS_ROOT)/herman/testn/storm.spm.log

herman.testh: herman.testh.tessa herman.testh.storm-add herman.testh.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/herman/testh/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/herman/testh/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/herman/testh/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/herman/testh/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/herman/testh/verify.csv
	plot --csv $(RESULTS_ROOT)/herman/testh/tessa.csv \
		$(RESULTS_ROOT)/herman/testh/storm.add.csv \
		$(RESULTS_ROOT)/herman/testh/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x horizon --y work_seconds elapsed_seconds elapsed_seconds \
		--title "Herman (N=17) — scaling with H" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/herman/testh/herman-testh.png
	plot --csv $(RESULTS_ROOT)/herman/testh/tessa.csv \
		$(RESULTS_ROOT)/herman/testh/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x horizon --y work_seconds measured_avg_seconds \
		--title "Herman (N=17) — scaling with H (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/herman/testh/herman-testh-tessa.png

herman.testh.tessa:
	test -f $(RESULTS_ROOT)/herman/testh/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/herman/testh/tessa.log --output-dir $(RESULTS_ROOT)/herman/testh herman -N $(HERMAN_TESTH_N) -H $(HERMAN_TESTH_H_TESSA)

herman.testh.storm-add:
	test -f $(RESULTS_ROOT)/herman/testh/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/herman/testh/storm.add.log --output-dir $(RESULTS_ROOT)/herman/testh herman -N $(HERMAN_TESTH_N) -H $(HERMAN_TESTH_H_STORM_ADD)

herman.testh.storm-spm:
	test -f $(RESULTS_ROOT)/herman/testh/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/herman/testh/storm.spm.log --output-dir $(RESULTS_ROOT)/herman/testh herman -N $(HERMAN_TESTH_N) -H $(HERMAN_TESTH_H_STORM_SPM)

clean.herman.testh.tessa:
	rm -f $(RESULTS_ROOT)/herman/testh/tessa.csv $(RESULTS_ROOT)/herman/testh/tessa.log
	rm -f $(RESULTS_ROOT)/herman/testh/*tessa*.time.jsonl

clean.herman.testh.storm-add:
	rm -f $(RESULTS_ROOT)/herman/testh/storm.add.csv $(RESULTS_ROOT)/herman/testh/storm.add.log

clean.herman.testh.storm-spm:
	rm -f $(RESULTS_ROOT)/herman/testh/storm.spm.csv $(RESULTS_ROOT)/herman/testh/storm.spm.log

# --- Meeting ---

meeting: meeting.testn meeting.testh

meeting.testn: meeting.testn.tessa meeting.testn.storm-add meeting.testn.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/meeting/testn/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/meeting/testn/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/meeting/testn/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/meeting/testn/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/meeting/testn/verify.csv
	plot --csv $(RESULTS_ROOT)/meeting/testn/tessa.csv \
		$(RESULTS_ROOT)/meeting/testn/storm.add.csv \
		$(RESULTS_ROOT)/meeting/testn/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x N --y work_seconds elapsed_seconds elapsed_seconds \
		--title "Meeting — scaling with N" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/meeting/testn/meeting-testn.png
	plot --csv $(RESULTS_ROOT)/meeting/testn/tessa.csv \
		$(RESULTS_ROOT)/meeting/testn/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x N --y work_seconds measured_avg_seconds \
		--title "Meeting — scaling with N (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/meeting/testn/meeting-testn-tessa.png

meeting.testn.tessa:
	test -f $(RESULTS_ROOT)/meeting/testn/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/meeting/testn/tessa.log --output-dir $(RESULTS_ROOT)/meeting/testn meeting -N $(MEETING_TESTN_N_TESSA) -H $(MEETING_TESTN_H)

meeting.testn.storm-add:
	test -f $(RESULTS_ROOT)/meeting/testn/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/meeting/testn/storm.add.log --output-dir $(RESULTS_ROOT)/meeting/testn meeting -N $(MEETING_TESTN_N_STORM_ADD) -H $(MEETING_TESTN_H)

meeting.testn.storm-spm:
	test -f $(RESULTS_ROOT)/meeting/testn/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/meeting/testn/storm.spm.log --output-dir $(RESULTS_ROOT)/meeting/testn meeting -N $(MEETING_TESTN_N_STORM_SPM) -H $(MEETING_TESTN_H)

clean.meeting.testn.tessa:
	rm -f $(RESULTS_ROOT)/meeting/testn/tessa.csv $(RESULTS_ROOT)/meeting/testn/tessa.log
	rm -f $(RESULTS_ROOT)/meeting/testn/*tessa*.time.jsonl

clean.meeting.testn.storm-add:
	rm -f $(RESULTS_ROOT)/meeting/testn/storm.add.csv $(RESULTS_ROOT)/meeting/testn/storm.add.log

clean.meeting.testn.storm-spm:
	rm -f $(RESULTS_ROOT)/meeting/testn/storm.spm.csv $(RESULTS_ROOT)/meeting/testn/storm.spm.log

meeting.testh: meeting.testh.tessa meeting.testh.storm-add meeting.testh.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/meeting/testh/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/meeting/testh/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/meeting/testh/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/meeting/testh/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/meeting/testh/verify.csv
	plot --csv $(RESULTS_ROOT)/meeting/testh/tessa.csv \
		$(RESULTS_ROOT)/meeting/testh/storm.add.csv \
		$(RESULTS_ROOT)/meeting/testh/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x horizon --y work_seconds elapsed_seconds elapsed_seconds \
		--title "Meeting (N=12) — scaling with H" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/meeting/testh/meeting-testh.png
	plot --csv $(RESULTS_ROOT)/meeting/testh/tessa.csv \
		$(RESULTS_ROOT)/meeting/testh/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x horizon --y work_seconds measured_avg_seconds \
		--title "Meeting (N=12) — scaling with H (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/meeting/testh/meeting-testh-tessa.png

meeting.testh.tessa:
	test -f $(RESULTS_ROOT)/meeting/testh/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/meeting/testh/tessa.log --output-dir $(RESULTS_ROOT)/meeting/testh meeting -N $(MEETING_TESTH_N) -H $(MEETING_TESTH_H_TESSA)

meeting.testh.storm-add:
	test -f $(RESULTS_ROOT)/meeting/testh/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/meeting/testh/storm.add.log --output-dir $(RESULTS_ROOT)/meeting/testh meeting -N $(MEETING_TESTH_N) -H $(MEETING_TESTH_H_STORM_ADD)

meeting.testh.storm-spm:
	test -f $(RESULTS_ROOT)/meeting/testh/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/meeting/testh/storm.spm.log --output-dir $(RESULTS_ROOT)/meeting/testh meeting -N $(MEETING_TESTH_N) -H $(MEETING_TESTH_H_STORM_SPM)

clean.meeting.testh.tessa:
	rm -f $(RESULTS_ROOT)/meeting/testh/tessa.csv $(RESULTS_ROOT)/meeting/testh/tessa.log
	rm -f $(RESULTS_ROOT)/meeting/testh/*tessa*.time.jsonl

clean.meeting.testh.storm-add:
	rm -f $(RESULTS_ROOT)/meeting/testh/storm.add.csv $(RESULTS_ROOT)/meeting/testh/storm.add.log

clean.meeting.testh.storm-spm:
	rm -f $(RESULTS_ROOT)/meeting/testh/storm.spm.csv $(RESULTS_ROOT)/meeting/testh/storm.spm.log

# --- Weather Factory ---

weather-factory: weather-factory.testn weather-factory.testh

weather-factory.testn: weather-factory.testn.tessa weather-factory.testn.storm-add weather-factory.testn.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/weather-factory/testn/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/weather-factory/testn/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/weather-factory/testn/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/weather-factory/testn/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/weather-factory/testn/verify.csv
	plot --csv $(RESULTS_ROOT)/weather-factory/testn/tessa.csv \
		$(RESULTS_ROOT)/weather-factory/testn/storm.add.csv \
		$(RESULTS_ROOT)/weather-factory/testn/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x N --y work_seconds elapsed_seconds elapsed_seconds \
		--title "Weather Factory — scaling with N" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/weather-factory/testn/weather-factory-testn.png
	plot --csv $(RESULTS_ROOT)/weather-factory/testn/tessa.csv \
		$(RESULTS_ROOT)/weather-factory/testn/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x N --y work_seconds measured_avg_seconds \
		--title "Weather Factory — scaling with N (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/weather-factory/testn/weather-factory-testn-tessa.png

weather-factory.testn.tessa:
	test -f $(RESULTS_ROOT)/weather-factory/testn/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/weather-factory/testn/tessa.log --output-dir $(RESULTS_ROOT)/weather-factory/testn weather-factory -N $(WF_TESTN_N_TESSA) -H $(WF_TESTN_H)

weather-factory.testn.storm-add:
	test -f $(RESULTS_ROOT)/weather-factory/testn/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/weather-factory/testn/storm.add.log --output-dir $(RESULTS_ROOT)/weather-factory/testn weather-factory -N $(WF_TESTN_N_STORM_ADD) -H $(WF_TESTN_H)

weather-factory.testn.storm-spm:
	test -f $(RESULTS_ROOT)/weather-factory/testn/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/weather-factory/testn/storm.spm.log --output-dir $(RESULTS_ROOT)/weather-factory/testn weather-factory -N $(WF_TESTN_N_STORM_SPM) -H $(WF_TESTN_H)

clean.weather-factory.testn.tessa:
	rm -f $(RESULTS_ROOT)/weather-factory/testn/tessa.csv $(RESULTS_ROOT)/weather-factory/testn/tessa.log
	rm -f $(RESULTS_ROOT)/weather-factory/testn/*tessa*.time.jsonl

clean.weather-factory.testn.storm-add:
	rm -f $(RESULTS_ROOT)/weather-factory/testn/storm.add.csv $(RESULTS_ROOT)/weather-factory/testn/storm.add.log

clean.weather-factory.testn.storm-spm:
	rm -f $(RESULTS_ROOT)/weather-factory/testn/storm.spm.csv $(RESULTS_ROOT)/weather-factory/testn/storm.spm.log

weather-factory.testh: weather-factory.testh.tessa weather-factory.testh.storm-add weather-factory.testh.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/weather-factory/testh/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/weather-factory/testh/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/weather-factory/testh/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/weather-factory/testh/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/weather-factory/testh/verify.csv
	plot --csv $(RESULTS_ROOT)/weather-factory/testh/tessa.csv \
		$(RESULTS_ROOT)/weather-factory/testh/storm.add.csv \
		$(RESULTS_ROOT)/weather-factory/testh/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x horizon --y work_seconds elapsed_seconds elapsed_seconds \
		--title "Weather Factory (N=13) — scaling with H" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/weather-factory/testh/weather-factory-testh.png
	plot --csv $(RESULTS_ROOT)/weather-factory/testh/tessa.csv \
		$(RESULTS_ROOT)/weather-factory/testh/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x horizon --y work_seconds measured_avg_seconds \
		--title "Weather Factory (N=13) — scaling with H (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/weather-factory/testh/weather-factory-testh-tessa.png

weather-factory.testh.tessa:
	test -f $(RESULTS_ROOT)/weather-factory/testh/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/weather-factory/testh/tessa.log --output-dir $(RESULTS_ROOT)/weather-factory/testh weather-factory -N $(WF_TESTH_N) -H $(WF_TESTH_H_TESSA)

weather-factory.testh.storm-add:
	test -f $(RESULTS_ROOT)/weather-factory/testh/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/weather-factory/testh/storm.add.log --output-dir $(RESULTS_ROOT)/weather-factory/testh weather-factory -N $(WF_TESTH_N) -H $(WF_TESTH_H_STORM_ADD)

weather-factory.testh.storm-spm:
	test -f $(RESULTS_ROOT)/weather-factory/testh/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/weather-factory/testh/storm.spm.log --output-dir $(RESULTS_ROOT)/weather-factory/testh weather-factory -N $(WF_TESTH_N) -H $(WF_TESTH_H_STORM_SPM)

clean.weather-factory.testh.tessa:
	rm -f $(RESULTS_ROOT)/weather-factory/testh/tessa.csv $(RESULTS_ROOT)/weather-factory/testh/tessa.log
	rm -f $(RESULTS_ROOT)/weather-factory/testh/*tessa*.time.jsonl

clean.weather-factory.testh.storm-add:
	rm -f $(RESULTS_ROOT)/weather-factory/testh/storm.add.csv $(RESULTS_ROOT)/weather-factory/testh/storm.add.log

clean.weather-factory.testh.storm-spm:
	rm -f $(RESULTS_ROOT)/weather-factory/testh/storm.spm.csv $(RESULTS_ROOT)/weather-factory/testh/storm.spm.log

# --- ParQueues ---

parqueues: parqueues.testq parqueues.testh

parqueues.testq: parqueues.testq.tessa parqueues.testq.storm-add parqueues.testq.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/parqueues/testq/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/parqueues/testq/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/parqueues/testq/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/parqueues/testq/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/parqueues/testq/verify.csv
	plot --csv $(RESULTS_ROOT)/parqueues/testq/tessa.csv \
		$(RESULTS_ROOT)/parqueues/testq/storm.add.csv \
		$(RESULTS_ROOT)/parqueues/testq/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x Q --y work_seconds elapsed_seconds elapsed_seconds \
		--xlabel "Q (queues)" \
		--title "ParQueues — scaling with Q" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/parqueues/testq/parqueues-testq.png
	plot --csv $(RESULTS_ROOT)/parqueues/testq/tessa.csv \
		$(RESULTS_ROOT)/parqueues/testq/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x Q --y work_seconds measured_avg_seconds \
		--xlabel "Q (queues)" \
		--title "ParQueues — scaling with Q (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/parqueues/testq/parqueues-testq-tessa.png

parqueues.testq.tessa:
	test -f $(RESULTS_ROOT)/parqueues/testq/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/parqueues/testq/tessa.log --output-dir $(RESULTS_ROOT)/parqueues/testq parqueues -Q $(PQ_TESTQ_Q_TESSA) -N $(PQ_TESTQ_N) -H $(PQ_TESTQ_H)

parqueues.testq.storm-add:
	test -f $(RESULTS_ROOT)/parqueues/testq/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/parqueues/testq/storm.add.log --output-dir $(RESULTS_ROOT)/parqueues/testq parqueues -Q $(PQ_TESTQ_Q_STORM_ADD) -N $(PQ_TESTQ_N) -H $(PQ_TESTQ_H)

parqueues.testq.storm-spm:
	test -f $(RESULTS_ROOT)/parqueues/testq/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/parqueues/testq/storm.spm.log --output-dir $(RESULTS_ROOT)/parqueues/testq parqueues -Q $(PQ_TESTQ_Q_STORM_SPM) -N $(PQ_TESTQ_N) -H $(PQ_TESTQ_H)

clean.parqueues.testq.tessa:
	rm -f $(RESULTS_ROOT)/parqueues/testq/tessa.csv $(RESULTS_ROOT)/parqueues/testq/tessa.log
	rm -f $(RESULTS_ROOT)/parqueues/testq/*tessa*.time.jsonl

clean.parqueues.testq.storm-add:
	rm -f $(RESULTS_ROOT)/parqueues/testq/storm.add.csv $(RESULTS_ROOT)/parqueues/testq/storm.add.log

clean.parqueues.testq.storm-spm:
	rm -f $(RESULTS_ROOT)/parqueues/testq/storm.spm.csv $(RESULTS_ROOT)/parqueues/testq/storm.spm.log

parqueues.testh: parqueues.testh.tessa parqueues.testh.storm-add parqueues.testh.storm-spm
	-$(VERIFIER) --log-file $(RESULTS_ROOT)/parqueues/testh/verify.log verify \
		--tessa-csv $(RESULTS_ROOT)/parqueues/testh/tessa.csv \
		--storm-csv $(RESULTS_ROOT)/parqueues/testh/storm.add.csv \
		--storm-csv $(RESULTS_ROOT)/parqueues/testh/storm.spm.csv \
		--output-csv $(RESULTS_ROOT)/parqueues/testh/verify.csv
	plot --csv $(RESULTS_ROOT)/parqueues/testh/tessa.csv \
		$(RESULTS_ROOT)/parqueues/testh/storm.add.csv \
		$(RESULTS_ROOT)/parqueues/testh/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x horizon --y work_seconds elapsed_seconds elapsed_seconds \
		--title "ParQueues — scaling with H" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/parqueues/testh/parqueues-testh.png
	plot --csv $(RESULTS_ROOT)/parqueues/testh/tessa.csv \
		$(RESULTS_ROOT)/parqueues/testh/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x horizon --y work_seconds measured_avg_seconds \
		--title "ParQueues — scaling with H (Tessa)" \
		--marker o --grid \
		--savefig $(RESULTS_ROOT)/parqueues/testh/parqueues-testh-tessa.png

parqueues.testh.tessa:
	test -f $(RESULTS_ROOT)/parqueues/testh/tessa.csv || \
		$(RUNNER_TESSA) --log-file $(RESULTS_ROOT)/parqueues/testh/tessa.log --output-dir $(RESULTS_ROOT)/parqueues/testh parqueues -Q $(PQ_TESTH_Q) -N $(PQ_TESTH_N) -H $(PQ_TESTH_H_TESSA)

parqueues.testh.storm-add:
	test -f $(RESULTS_ROOT)/parqueues/testh/storm.add.csv || \
		$(RUNNER_STORM_ADD) --log-file $(RESULTS_ROOT)/parqueues/testh/storm.add.log --output-dir $(RESULTS_ROOT)/parqueues/testh parqueues -Q $(PQ_TESTH_Q) -N $(PQ_TESTH_N) -H $(PQ_TESTH_H_STORM_ADD)

parqueues.testh.storm-spm:
	test -f $(RESULTS_ROOT)/parqueues/testh/storm.spm.csv || \
		$(RUNNER_STORM_SPM) --log-file $(RESULTS_ROOT)/parqueues/testh/storm.spm.log --output-dir $(RESULTS_ROOT)/parqueues/testh parqueues -Q $(PQ_TESTH_Q) -N $(PQ_TESTH_N) -H $(PQ_TESTH_H_STORM_SPM)

clean.parqueues.testh.tessa:
	rm -f $(RESULTS_ROOT)/parqueues/testh/tessa.csv $(RESULTS_ROOT)/parqueues/testh/tessa.log
	rm -f $(RESULTS_ROOT)/parqueues/testh/*tessa*.time.jsonl

clean.parqueues.testh.storm-add:
	rm -f $(RESULTS_ROOT)/parqueues/testh/storm.add.csv $(RESULTS_ROOT)/parqueues/testh/storm.add.log

clean.parqueues.testh.storm-spm:
	rm -f $(RESULTS_ROOT)/parqueues/testh/storm.spm.csv $(RESULTS_ROOT)/parqueues/testh/storm.spm.log

# --- Tessa-only workflow (skips Storm; assumes Storm CSVs exist) ---
# Use when iterating on Tessa without rerunning Storm baselines. The verify
# and plot steps read storm.{add,spm}.csv from disk; the Storm targets are
# NOT prerequisites, so a missing Storm CSV will surface as a verify/plot
# error rather than silently trigger a rebuild.
#
# Usage:
#   make -f reproduce.mk tessa.all           # full Tessa sweep + verify + plots
#   make -f reproduce.mk SMOKE=1 tessa.all   # quick smoke check
#   make -f reproduce.mk clean.tessa.all     # remove Tessa CSVs + verify + plots

# $(call _tessa_all_rules,SUITE,PHASE,X_AXIS,TITLE,IMG_STEM,EXTRA_PLOT_ARGS)
# Defines <SUITE>.<PHASE>.tessa.all and clean.<SUITE>.<PHASE>.tessa.all.
define _tessa_all_rules
$(1).$(2).tessa.all: $(1).$(2).tessa
	-$$(VERIFIER) --log-file $$(RESULTS_ROOT)/$(1)/$(2)/verify.log verify \
		--tessa-csv $$(RESULTS_ROOT)/$(1)/$(2)/tessa.csv \
		--storm-csv $$(RESULTS_ROOT)/$(1)/$(2)/storm.add.csv \
		--storm-csv $$(RESULTS_ROOT)/$(1)/$(2)/storm.spm.csv \
		--output-csv $$(RESULTS_ROOT)/$(1)/$(2)/verify.csv
	plot --csv $$(RESULTS_ROOT)/$(1)/$(2)/tessa.csv \
		$$(RESULTS_ROOT)/$(1)/$(2)/storm.add.csv \
		$$(RESULTS_ROOT)/$(1)/$(2)/storm.spm.csv \
		--label "Tessa" "Storm (ADD)" "Storm (SPM)" \
		--x $(3) --y work_seconds elapsed_seconds elapsed_seconds $(6) \
		--title "$(4)" \
		--marker o --grid \
		--savefig $$(RESULTS_ROOT)/$(1)/$(2)/$(5).png
	plot --csv $$(RESULTS_ROOT)/$(1)/$(2)/tessa.csv \
		$$(RESULTS_ROOT)/$(1)/$(2)/tessa.csv \
		--label "Tessa 1st" "Tessa 2nd" \
		--x $(3) --y work_seconds measured_avg_seconds $(6) \
		--title "$(4) (Tessa)" \
		--marker o --grid \
		--savefig $$(RESULTS_ROOT)/$(1)/$(2)/$(5)-tessa.png

clean.$(1).$(2).tessa.all: clean.$(1).$(2).tessa
	rm -f $$(RESULTS_ROOT)/$(1)/$(2)/verify.csv $$(RESULTS_ROOT)/$(1)/$(2)/verify.log
	rm -f $$(RESULTS_ROOT)/$(1)/$(2)/$(5).png $$(RESULTS_ROOT)/$(1)/$(2)/$(5)-tessa.png
endef

$(eval $(call _tessa_all_rules,herman,testn,N,Herman — scaling with N,herman-testn,))
$(eval $(call _tessa_all_rules,herman,testh,horizon,Herman (N=17) — scaling with H,herman-testh,))
$(eval $(call _tessa_all_rules,meeting,testn,N,Meeting — scaling with N,meeting-testn,))
$(eval $(call _tessa_all_rules,meeting,testh,horizon,Meeting (N=12) — scaling with H,meeting-testh,))
$(eval $(call _tessa_all_rules,weather-factory,testn,N,Weather Factory — scaling with N,weather-factory-testn,))
$(eval $(call _tessa_all_rules,weather-factory,testh,horizon,Weather Factory (N=13) — scaling with H,weather-factory-testh,))
$(eval $(call _tessa_all_rules,parqueues,testq,Q,ParQueues — scaling with Q,parqueues-testq,--xlabel "Q (queues)"))
$(eval $(call _tessa_all_rules,parqueues,testh,horizon,ParQueues — scaling with H,parqueues-testh,))

tessa.all: herman.testn.tessa.all herman.testh.tessa.all \
	meeting.testn.tessa.all meeting.testh.tessa.all \
	weather-factory.testn.tessa.all weather-factory.testh.tessa.all \
	parqueues.testq.tessa.all parqueues.testh.tessa.all

clean.tessa.all: clean.herman.testn.tessa.all clean.herman.testh.tessa.all \
	clean.meeting.testn.tessa.all clean.meeting.testh.tessa.all \
	clean.weather-factory.testn.tessa.all clean.weather-factory.testh.tessa.all \
	clean.parqueues.testq.tessa.all clean.parqueues.testh.tessa.all

.PHONY: tessa.all clean.tessa.all \
	herman.testn.tessa.all herman.testh.tessa.all \
	meeting.testn.tessa.all meeting.testh.tessa.all \
	weather-factory.testn.tessa.all weather-factory.testh.tessa.all \
	parqueues.testq.tessa.all parqueues.testh.tessa.all \
	clean.herman.testn.tessa.all clean.herman.testh.tessa.all \
	clean.meeting.testn.tessa.all clean.meeting.testh.tessa.all \
	clean.weather-factory.testn.tessa.all clean.weather-factory.testh.tessa.all \
	clean.parqueues.testq.tessa.all clean.parqueues.testh.tessa.all
