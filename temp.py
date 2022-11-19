# libraries
from tensorflow import convert_to_tensor
from spektral.models import GCN
from spektral.data import MixedLoader
from lorenz import lorenzDataset, DEFAULT_TIME_RESOLUTION

dataset = lorenzDataset(
    n_samples=2,
    # input_steps=2 / DEFAULT_TIME_RESOLUTION,  # 2 days
    # output_delay=1 / DEFAULT_TIME_RESOLUTION,  # 1 day
    # output_steps=1,
    min_buffer=-3 / DEFAULT_TIME_RESOLUTION,
    # rand_buffer=False,
    # K=36,
    # F=8,
    # c=10,
    # b=10,
    # h=1,
    # coupled=True,
    # time_resolution=DEFAULT_TIME_RESOLUTION,
    # seed=42
)

data_loader = MixedLoader(dataset=dataset, batch_size=2, shuffle=False)

GCN_model = GCN(n_labels=1)
GCN_model.compile(optimizer="adam", loss="mean_squared_error")
GCN_model.fit(data_loader.load(),
                            steps_per_epoch=data_loader.steps_per_epoch,
                            epochs=1)
# GCN_model.build()


print(GCN_model.summary())