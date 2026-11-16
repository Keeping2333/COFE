---                                                                                                                                      
  Full-Stack RCA - Complete File Reference & Usage Guide                                                                                   
                                                                                                                                           
  Overview                                                                                                                                 

  This is a multimodal root cause analysis system for microservices with two stages:                                                       
  1. Stage 1: Contrastive learning for anomaly detection                                                                                   
  2. Stage 2: LLM-based reasoning with SOP validation                                                                                      
                                                                                                                                           
  ---                                                                                                                                      
  Installation & Setup

  pyproject.toml

  Purpose: Project metadata and dependencies definition

  How to use:
  cd rca

  # Install in development mode
  pip install -e .

  # Install with dev tools (pytest, black, mypy)
  pip install -e ".[dev]"

  # Install with Milvus support
  pip install -e ".[milvus]"

  requirements.txt

  Purpose: Flat list of dependencies (alternative to pyproject.toml)

  How to use:
  pip install -r requirements.txt

  setup.py

  Purpose: Legacy setuptools configuration for editable installs

  How to use:
  pip install -e .

  ---
  Source Code (src/fullstack_rca/)

  __init__.py

  Purpose: Package entry point, exports FaultLayer enum

  How to use:
  from fullstack_rca import FaultLayer, __version__

  print(__version__)  # "0.1.0"
  print(FaultLayer.L1_INFRASTRUCTURE)  # Layer enum

  ---
  Data Module (src/fullstack_rca/data/)

  dataset.py

  Purpose: Trace-centric multimodal dataset loader for training

  Key classes:
  - MultimodalSample: Single aligned sample (trace + logs + metrics)
  - MicroServiceDataset: PyTorch Dataset for batch loading

  How to use:
  from fullstack_rca.data.dataset import MicroServiceDataset, collate_multimodal_samples
  from torch.utils.data import DataLoader

  # Create dataset from experiment directory
  dataset = MicroServiceDataset(
      experiment_dir="/path/to/experiment",
      trace_dir="traces",
      log_file="logs/all_logs.json",
      metric_dir="metrics",
      time_window_seconds=60.0,  # metric alignment window
      max_logs_per_trace=50,
      max_metrics=20
  )

  # Create DataLoader
  loader = DataLoader(
      dataset,
      batch_size=32,
      shuffle=True,
      collate_fn=collate_multimodal_samples
  )

  # Iterate
  for batch in loader:
      traces = batch['traces']      # [batch, num_spans, span_features]
      logs = batch['log_tokens']    # [batch, num_logs, max_len]
      metrics = batch['metrics']    # [batch, num_metrics, seq_len]
      positions = batch['relative_positions']  # for Gaussian attention

  alignment.py

  Purpose: Hard/soft alignment utilities for multimodal data

  Key functions:
  - align_logs_by_trace_id(): HARD alignment - exact trace_id match
  - align_metrics_by_time_window(): SOFT alignment - time-based
  - compute_relative_time_positions(): For Gaussian attention weights
  - build_trace_graph(): Construct adjacency matrix from spans

  How to use:
  from fullstack_rca.data.alignment import (
      align_logs_by_trace_id,
      align_metrics_by_time_window,
      compute_relative_time_positions,
      build_trace_graph
  )

  # Align logs to a trace (HARD alignment)
  aligned_logs = align_logs_by_trace_id(
      trace_id="abc123def456789012345678",
      logs=log_entries  # List[LogEntry]
  )

  # Align metrics by time window (SOFT alignment)
  aligned_metrics, rel_positions = align_metrics_by_time_window(
      trace_start_time=1700000000.0,
      trace_end_time=1700000000.05,
      metrics=metric_series,
      window_seconds=60.0
  )

  # Build trace graph adjacency
  adjacency = build_trace_graph(spans)  # [num_spans, num_spans]

  ---
  Parsers (src/fullstack_rca/data/parsers/)

  jaeger.py

  Purpose: Parse Jaeger distributed trace JSON

  Key classes:
  - Span: Single span with timing, tags, service info
  - Trace: Complete trace with all spans

  How to use:
  from fullstack_rca.data.parsers.jaeger import JaegerParser, Span, Trace

  parser = JaegerParser()

  # Parse from file
  traces = parser.parse_file("/path/to/traces.json")

  # Access trace data
  for trace in traces:
      print(f"Trace ID: {trace.trace_id}")
      print(f"Services: {trace.services}")
      print(f"Duration: {trace.duration_ms}ms")
      for span in trace.spans:
          print(f"  {span.service_name}/{span.operation_name}: {span.duration_us}μs")

  loki.py

  Purpose: Parse Loki log stream JSON

  Key classes:
  - LogEntry: Single log with timestamp, trace_id, severity
  - LogStream: Stream of logs from one source

  How to use:
  from fullstack_rca.data.parsers.loki import LokiParser, LogEntry

  parser = LokiParser()

  # Parse from file
  log_streams = parser.parse_file("/path/to/logs.json")

  # Access logs
  for stream in log_streams:
      print(f"Source: {stream.labels}")
      for entry in stream.entries:
          print(f"  [{entry.severity}] {entry.body}")
          print(f"    trace_id: {entry.trace_id}")

  prometheus.py

  Purpose: Parse Prometheus PromQL matrix results

  Key classes:
  - MetricSeries: Time series with metric name, labels, values

  How to use:
  from fullstack_rca.data.parsers.prometheus import PrometheusParser

  parser = PrometheusParser()

  # Parse from file
  metrics = parser.parse_file("/path/to/metrics.json")

  # Access metric series
  for series in metrics:
      print(f"Metric: {series.metric_name}")
      print(f"Service: {series.service_name}")
      print(f"Values: {list(zip(series.timestamps, series.values))}")

  ground_truth.py

  Purpose: Parse ground truth labels from experiment metadata

  How to use:
  from fullstack_rca.data.parsers.ground_truth import parse_ground_truth

  # Parse from meta.json
  ground_truth = parse_ground_truth("/path/to/experiment/meta.json")
  print(f"Fault type: {ground_truth.fault_type}")
  print(f"Layer: {ground_truth.layer}")
  print(f"Affected services: {ground_truth.affected_services}")

  ---
  Workload Simulation (src/fullstack_rca/data/workload/)

  state_machine.py

  Purpose: FSM for realistic user behavior simulation in load tests

  Key classes:
  - UserState: State enum (BROWSING, SEARCHING, BOOKED, etc.)
  - UserBehaviorFSM: Probabilistic state machine

  How to use:
  from fullstack_rca.data.workload.state_machine import UserBehaviorFSM, UserState

  # Create FSM
  fsm = UserBehaviorFSM()

  # Simulate user session
  current_state = UserState.INITIAL
  while current_state != UserState.TERMINAL:
      action = fsm.get_action(current_state)
      print(f"State: {current_state}, Action: {action}")
      current_state = fsm.transition(current_state)

  ---
  Models (src/fullstack_rca/models/)

  base.py

  Purpose: Abstract base classes for all encoders

  Key classes:
  - FaultLayer: Enum (L1_INFRASTRUCTURE, L2_NETWORK, L3_PLATFORM, L4_APPLICATION)
  - EncoderOutput: Standard output format for encoders
  - BaseEncoder: Abstract encoder with projection head
  - LayerAwareEncoder: Encoder with 4-layer classification head

  How to use:
  from fullstack_rca.models.base import FaultLayer, EncoderOutput, LayerAwareEncoder

  # FaultLayer for classification
  layer = FaultLayer.L1_INFRASTRUCTURE
  print(layer.value)  # 0

  # Creating custom encoder
  class MyEncoder(LayerAwareEncoder):
      @property
      def modality(self) -> str:
          return "custom"

      def encode(self, x, mask=None):
          # Implementation
          pass

  ---
  Encoders (src/fullstack_rca/models/encoders/)

  log_encoder.py

  Purpose: Encode log messages using DistilBERT

  How to use:
  from fullstack_rca.models.encoders.log_encoder import LogEncoder
  import torch

  # Create encoder
  encoder = LogEncoder(
      output_dim=256,
      pretrained_model="distilbert-base-uncased",
      use_drain3=False,  # Set True for template extraction
      freeze_bert=False
  )

  # Encode logs
  log_tokens = torch.randint(0, 30522, (batch_size, num_logs, max_len))
  log_mask = torch.ones(batch_size, num_logs, max_len)

  output = encoder(log_tokens, log_mask)
  print(output.pooled.shape)  # [batch_size, 256]
  print(output.layer_assignments.shape)  # [batch_size, 4]

  trace_encoder.py

  Purpose: Encode traces as graphs using Graph Attention Networks (GAT)

  How to use:
  from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
  import torch

  # Create encoder
  encoder = TraceEncoder(
      input_dim=64,       # span feature dimension
      hidden_dim=128,
      output_dim=256,
      num_heads=4,
      num_layers=3,
      dropout=0.1
  )

  # Encode traces
  span_features = torch.randn(batch_size, num_spans, 64)
  adjacency = torch.zeros(batch_size, num_spans, num_spans)  # parent-child edges

  output = encoder(span_features, adjacency)
  print(output.pooled.shape)  # [batch_size, 256]

  gaussian_attention.py

  Purpose: Time-aware Gaussian attention for soft metric alignment

  How to use:
  from fullstack_rca.models.encoders.gaussian_attention import GaussianPositionalEncoding
  import torch

  # Create module
  gaussian = GaussianPositionalEncoding(
      initial_sigma=10.0,  # learnable bandwidth
      learnable=True
  )

  # Compute attention weights
  relative_positions = torch.abs(metric_times - trace_time)  # |t_metric - t_trace|
  weights = gaussian(relative_positions)  # [batch, seq_len]
  # Weights are higher for metrics closer in time to the trace

  ---
  Fusion (src/fullstack_rca/models/fusion/)

  projector.py

  Purpose: Fuse multimodal embeddings into unified 4-layer space

  How to use:
  from fullstack_rca.models.fusion.projector import MultiModalProjector
  import torch

  # Create projector
  projector = MultiModalProjector(
      log_dim=256,
      trace_dim=256,
      metric_dim=256,
      output_dim=512,
      fusion_method="attention",  # or "concat", "gated"
      num_layers=4
  )

  # Fuse modalities
  log_emb = torch.randn(batch_size, 256)
  trace_emb = torch.randn(batch_size, 256)
  metric_emb = torch.randn(batch_size, 256)

  fused = projector(log_emb, trace_emb, metric_emb)
  print(fused.shape)  # [batch_size, 512]

  ---
  Losses (src/fullstack_rca/models/losses/)

  contrastive.py

  Purpose: Layer-aware contrastive loss for training

  How to use:
  from fullstack_rca.models.losses.contrastive import LayerContrastiveLoss
  import torch

  # Create loss function
  loss_fn = LayerContrastiveLoss(
      temperature=0.07,
      use_hard_negatives=True,
      margin=0.5
  )

  # Compute loss
  embeddings = torch.randn(batch_size, 256)
  layer_labels = torch.randint(0, 4, (batch_size,))  # L1-L4

  loss = loss_fn(embeddings, layer_labels)
  loss.backward()

  ---
  RAG Module (src/fullstack_rca/rag/)

  vectorstore/base.py

  Purpose: Abstract vector store interface for similarity search

  How to use:
  from fullstack_rca.rag.vectorstore.base import VectorStore, Document, SearchResult

  # Create vector store (FAISS backend)
  store = VectorStore.create(
      backend="faiss",
      embedding_dim=256,
      index_type="IVF_FLAT",
      metric="cosine"
  )

  # Add documents
  doc = Document(
      id="episode_001",
      embedding=embedding_vector,
      text="CPU spike on ts-order-service",
      metadata={"layer": "L1", "fault_type": "cpu_pressure"}
  )
  store.add([doc])

  # Search similar
  results = store.search(query_embedding, k=5)
  for r in results:
      print(f"ID: {r.id}, Score: {r.score}")

  knowledge/episode.py

  Purpose: Represent historical fault episodes

  How to use:
  from fullstack_rca.rag.knowledge.episode import Episode, TelemetrySummary

  # Create episode
  episode = Episode(
      id="ep_001",
      experiment_id="exp_cpu_001",
      embedding=embedding_vector,
      layer=FaultLayer.L1_INFRASTRUCTURE,
      fault_type="cpu_pressure",
      affected_services=["ts-order-service"],
      description="High CPU utilization caused request timeouts",
      telemetry_summary=TelemetrySummary(
          metrics=["node_cpu_seconds_total > 90%"],
          logs=["Connection timeout", "Request failed"],
          traces=["High latency in /api/orders"]
      ),
      time_window=(start_time, end_time)
  )

  knowledge/manager.py

  Purpose: Manage knowledge base of historical episodes

  How to use:
  from fullstack_rca.rag.knowledge.manager import KnowledgeManager

  # Create manager
  manager = KnowledgeManager(
      vector_store=store,
      metadata_path="/path/to/metadata.json"
  )

  # Add episode
  manager.add_episode(episode)

  # Retrieve similar episodes for LLM context
  similar = manager.retrieve(
      query_embedding=current_incident_embedding,
      k=3
  )
  for ep in similar:
      print(f"Similar: {ep.description}")

  ---
  Agent Module (src/fullstack_rca/agent/)

  base.py

  Purpose: Abstract LLM reasoning agent framework

  Key classes:
  - DiagnosisConfidence: HIGH/MEDIUM/LOW enum
  - LayerEvidence: Evidence collected per layer
  - DiagnosisResult: Final diagnosis output
  - AgentContext: Input context for reasoning
  - ReasoningAgent: Abstract agent base class

  How to use:
  from fullstack_rca.agent.base import ReasoningAgent, AgentContext, DiagnosisResult

  # Create context for agent
  context = AgentContext(
      embeddings=fused_embeddings,
      telemetry_summaries={
          "metrics": ["CPU: 95%", "Memory: 80%"],
          "logs": ["Error: Connection timeout"],
          "traces": ["Latency spike in order-service"]
      },
      rag_episodes=similar_episodes,
      incident_scope=["ts-order-service", "ts-order-mongo"]
  )

  # Run diagnosis (concrete agent implementation required)
  agent = MyReasoningAgent(llm_client=anthropic_client)
  result: DiagnosisResult = agent.diagnose(context)

  print(f"Root cause: {result.root_cause_layer}")
  print(f"Type: {result.fault_type}")
  print(f"Confidence: {result.confidence}")
  print(f"Reasoning: {result.reasoning_trace}")

  sop/validator.py

  Purpose: Validate LLM diagnosis against SOP rules

  How to use:
  from fullstack_rca.agent.sop.validator import SOPValidator, SOPRule

  # Create validator
  validator = SOPValidator()

  # Validate diagnosis
  is_valid, violations = validator.validate(diagnosis_result)

  if not is_valid:
      print("SOP Violations:")
      for v in violations:
          print(f"  - {v.rule_name}: {v.message}")

      # Auto-correct
      corrected = validator.correct(diagnosis_result, violations)

  ---
  Tests (tests/)

  conftest.py

  Purpose: Pytest fixtures for testing

  How to use:
  # Run all tests
  cd rca
  pytest

  # Run with coverage
  pytest --cov=fullstack_rca --cov-report=html

  # Run specific test file
  pytest tests/test_dataset.py -v

  # Run tests matching pattern
  pytest -k "alignment" -v

  Available fixtures:
  - sample_trace_data: Jaeger format test data
  - sample_log_data: Loki format test data
  - sample_metric_data: Prometheus format test data
  - temp_experiment_dir: Complete experiment directory
  - device: CPU/GPU device
  - sample_embeddings, sample_layer_labels: Tensor fixtures

  ---
  Placeholder Directories (To Be Implemented)
  ┌──────────────────────────────┬────────────────────────────────────┐
  │          Directory           │          Intended Purpose          │
  ├──────────────────────────────┼────────────────────────────────────┤
  │ configs/                     │ Hydra/YAML configuration files     │
  ├──────────────────────────────┼────────────────────────────────────┤
  │ notebooks/                   │ Jupyter notebooks for analysis     │
  ├──────────────────────────────┼────────────────────────────────────┤
  │ scripts/                     │ CLI scripts for training/inference │
  ├──────────────────────────────┼────────────────────────────────────┤
  │ src/fullstack_rca/training/  │ Training loops and optimizers      │
  ├──────────────────────────────┼────────────────────────────────────┤
  │ src/fullstack_rca/inference/ │ Production inference pipelines     │
  ├──────────────────────────────┼────────────────────────────────────┤
  │ src/fullstack_rca/utils/     │ Helper functions                   │
  └──────────────────────────────┴────────────────────────────────────┘
  ---
  Complete Training Pipeline (Example)

  import torch
  from torch.utils.data import DataLoader
  from fullstack_rca.data.dataset import MicroServiceDataset, collate_multimodal_samples
  from fullstack_rca.models.encoders.log_encoder import LogEncoder
  from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
  from fullstack_rca.models.fusion.projector import MultiModalProjector
  from fullstack_rca.models.losses.contrastive import LayerContrastiveLoss

  # 1. Load data
  dataset = MicroServiceDataset(experiment_dir="/path/to/exp")
  loader = DataLoader(dataset, batch_size=32, collate_fn=collate_multimodal_samples)

  # 2. Create models
  log_encoder = LogEncoder(output_dim=256)
  trace_encoder = TraceEncoder(input_dim=64, output_dim=256)
  projector = MultiModalProjector(log_dim=256, trace_dim=256, metric_dim=256, output_dim=512)
  loss_fn = LayerContrastiveLoss(temperature=0.07)

  # 3. Training loop
  optimizer = torch.optim.AdamW(
      list(log_encoder.parameters()) +
      list(trace_encoder.parameters()) +
      list(projector.parameters()),
      lr=1e-4
  )

  for epoch in range(num_epochs):
      for batch in loader:
          log_emb = log_encoder(batch['log_tokens'], batch['log_mask']).pooled
          trace_emb = trace_encoder(batch['traces'], batch['adjacency']).pooled
          metric_emb = ...  # metric encoder

          fused = projector(log_emb, trace_emb, metric_emb)
          loss = loss_fn(fused, batch['layer_labels'])

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

  ---
  Data Directory Structure Expected

  experiment_001/
  ├── meta.json              # Ground truth and experiment metadata
  ├── traces/
  │   └── traces.json        # Jaeger format
  ├── logs/
  │   └── all_logs.json      # Loki format
  └── metrics/
      └── metrics.json       # Prometheus format

  ---