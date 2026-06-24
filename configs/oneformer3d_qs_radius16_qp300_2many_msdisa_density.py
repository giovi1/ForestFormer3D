_base_ = './oneformer3d_qs_radius16_qp300_2many.py'

model = dict(
    query_selection=dict(
        mode='density',
        num_queries=300,
        isa_ratio=0.5,
        multiscale_ratio=0.0,
        density_ratio=0.5,
        density_sampling_mode='balanced',
        density_deterministic=True,
        density_k=16,
        density_eps=1e-6,
        deduplicate_queries=True,
        fallback_to_isa=True,
        log_interval=100))
