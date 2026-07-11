from utils.config import (
    get_config,
    print_summary,
    validate_paths,
)

cfg = get_config()

validate_paths(cfg)

print_summary(cfg)

print()

print(cfg.dataset.root)
print(cfg.model.hidden_dim)
print(cfg.training.learning_rate)
print(cfg.loss.score_weight)
print(cfg.runtime.device)
print(cfg.cache.enabled)
