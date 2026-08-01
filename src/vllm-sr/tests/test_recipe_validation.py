from cli.models import UserConfig
from cli.validator import validate_user_config


def recipe_config(*, recipe_model: str = "model-a", recipe_name: str = "private"):
    return UserConfig.model_validate(
        {
            "version": "v0.3",
            "providers": {
                "defaults": {"default_model": "model-a"},
                "models": [
                    {
                        "name": "model-a",
                        "backend_refs": [{"endpoint": "127.0.0.1:8000"}],
                    }
                ],
            },
            "routing": {
                "modelCards": [{"name": "model-a"}],
                "decisions": [
                    {
                        "name": "default-route",
                        "description": "default",
                        "priority": 1,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": "model-a"}],
                    }
                ],
            },
            "entrypoints": [
                {"model_names": ["amd/rocm-v1-private"], "recipe": recipe_name}
            ],
            "recipes": [
                {
                    "name": "private",
                    "routing": {
                        "decisions": [
                            {
                                "name": "private-route",
                                "description": "private",
                                "priority": 1,
                                "rules": {"operator": "AND", "conditions": []},
                                "modelRefs": [{"model": recipe_model}],
                            }
                        ]
                    },
                }
            ],
        }
    )


def test_recipe_model_references_are_validated():
    errors = validate_user_config(recipe_config(recipe_model="missing-model"))

    assert any("unknown model 'missing-model'" in error.message for error in errors)


def test_entrypoints_must_reference_known_recipe():
    errors = validate_user_config(recipe_config(recipe_name="missing-recipe"))

    assert any("unknown recipe 'missing-recipe'" in error.message for error in errors)


def test_entrypoint_names_cannot_collide_with_provider_models():
    config = recipe_config()
    config.entrypoints[0].model_names = ["model-a"]

    errors = validate_user_config(config)

    assert any("conflicts with a configured model" in error.message for error in errors)


def test_recipes_only_default_profile_is_allowed():
    config = recipe_config()
    config.routing.decisions = []
    config.recipes[0].name = "default"
    config.entrypoints[0].recipe = "default"

    errors = validate_user_config(config)

    assert not any(
        "Duplicate recipe name 'default'" in error.message for error in errors
    )
    assert not any("unknown recipe 'default'" in error.message for error in errors)
