# Train one model for each task

from gail import gail_train
from behavioural_cloning import  behavioural_cloning_train
if __name__ == "__main__":
    print("===Training FindCave model===")
    # behavioural_cloning_train(
    #     name='MineRLBasaltFindCave-v0',
    #     data_dir="data/MineRLBasaltFindCave-v0",
    #     in_model="data/VPT-models/foundation-model-1x.model",
    #     in_weights="data/VPT-models/foundation-model-1x.weights",
    #     out_weights="train/MineRLBasaltFindCave"
    # )
    gail_train(
        name='MineRLBasaltFindCave-v0',
        data_dir="data/MineRLBasaltFindCave-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindCave"
    )
    print("===Training MakeWaterfall model===")
    gail_train(
        name='MineRLBasaltMakeWaterfall-v0',
        data_dir="data/MineRLBasaltMakeWaterfall-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltMakeWaterfall.weights"
    )

    print("===Training CreateVillageAnimalPen model===")
    gail_train(
        name='MineRLBasaltCreateVillageAnimalPen-v0',
        data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    )

    print("===Training BuildVillageHouse model===")
    gail_train(
        name='MineRLBasaltBuildVillageHouse-v0',
        data_dir="data/MineRLBasaltBuildVillageHouse-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    )