from torch.utils.data import DataLoader
from prompt_engineering.finetune.finetune import DataModule


def test_train_dataloader():
    # Create an instance of the DataModule class
    data_module = DataModule(model_id="model", dir_data="./data", batch_size=32, batch_size_inf=16,
                             train_dset_names=["research_paper", "wizardLM_instruct_70k"], val_tasks=["next_token", "spot_alignment"],
                             num_workers=4, seed=10)

    # Call the train_dataloader method
    train_dataloader = data_module.train_dataloader()

    # Assert that the train_dataloader is not None
    assert train_dataloader is not None

    # Assert that the train_dataloader is an instance of DataLoader
    assert isinstance(train_dataloader, DataLoader)

    # Add more assertions if necessary


def test_val_dataloader():
    # Create an instance of the DataModule class
    data_module = DataModule(model_id="model1", dir_data="./data", batch_size=32, batch_size_inf=16,
                             train_dset_names=["research_paper", "wizardLM_instruct_70k"], val_tasks=["next_token", "spot_alignment"],
                             num_workers=4, seed=10)

    # Call the val_dataloader method
    val_dataloader = data_module.val_dataloader()

    # Assert that the val_dataloader is not None
    assert val_dataloader is not None

    # Assert that the val_dataloader is a list
    assert isinstance(val_dataloader, list)

    # Assert that each item in the val_dataloader list is an instance of DataLoader
    for dataloader in val_dataloader:
        assert isinstance(dataloader, DataLoader)

    # Add more assertions if necessary


def test_test_dataloader():
    # Create an instance of the DataModule class
    data_module = DataModule(model_id="model1", dir_data="./data", batch_size=32, batch_size_inf=16,
                             train_dset_names=["research_paper", "wizardLM_instruct_70k"], val_tasks=["next_token", "spot_alignment"],
                             num_workers=4, seed=10)

    # Call the test_dataloader method
    test_dataloader = data_module.test_dataloader()

    # Assert that the test_dataloader is not None
    assert test_dataloader is not None

    # Assert that the test_dataloader is an instance of DataLoader
    assert isinstance(test_dataloader, DataLoader)

    # Add more assertions if necessary
