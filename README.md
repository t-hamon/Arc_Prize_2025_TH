# Arc_Prize_2025_TH
ARC Challenge involving the prediction of specific tasks. The grids maintain their input dimensions.


The data is imported into the "data" folder. Some of it is imported from the ARC Prize Challenge, and the following files were generated:
- detected_transformations.json
- predicted_solutions.json
- task_classification.json
- transformation_catalog.json (and .csv)

The data/arc_utils.py script loads the initial data and saves the detected transformations. The scripts in the analysis folder also retrieve all the detected transformations.

The transformations.py script implements functions for specific transformations and applies series of these transformations to a given grid, then repeats the grid according to a scaling factor.

It is important to emphasize that I did not perform the various resizings of the grids between input/output.

The detected transformations are then listed in the "transformation_catalog" via the create_transformation_catalog.ipynb script.

"analysis_transformations.ipynb" was created to attempt solutions

The data is then listed as follows:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task_id</th>
      <th>pair_id</th>
      <th>input_shape</th>
      <th>output_shape</th>
      <th>input_colors</th>
      <th>output_colors</th>
      <th>recoloration</th>
      <th>rotation_90</th>
      <th>rotation_180</th>
      <th>rotation_270</th>
      <th>...</th>
      <th>repetition</th>
      <th>padding</th>
      <th>translation</th>
      <th>tiled_subgrid</th>
      <th>partial_symetry</th>
      <th>pattern_checkerboard</th>
      <th>pattern_horizontal_stripes</th>
      <th>pattern_vertical_stripes</th>
      <th>pattern_uniform_blocks</th>
      <th>shape_based_recoloration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00576224</td>
      <td>0</td>
      <td>[2, 2]</td>
      <td>[6, 6]</td>
      <td>[3, 4, 7, 9]</td>
      <td>[3, 4, 7, 9]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>None</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00576224</td>
      <td>1</td>
      <td>[2, 2]</td>
      <td>[6, 6]</td>
      <td>[4, 6, 8]</td>
      <td>[4, 6, 8]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>None</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>007bbfb7</td>
      <td>0</td>
      <td>[3, 3]</td>
      <td>[9, 9]</td>
      <td>[0, 6]</td>
      <td>[0, 6]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>None</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>007bbfb7</td>
      <td>1</td>
      <td>[3, 3]</td>
      <td>[9, 9]</td>
      <td>[0, 4]</td>
      <td>[0, 4]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>None</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>007bbfb7</td>
      <td>2</td>
      <td>[3, 3]</td>
      <td>[9, 9]</td>
      <td>[0, 2]</td>
      <td>[0, 2]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>None</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
Tensors are created with PyTorch, then we use a ConvNet, convert to one-hot labels per pixel, and then train the ConvNet model for 40 Epochs.

First Problem:
The grid backgrounds are black. Since in the entire dataset, there is much more black than any other color, the model predicts black almost everywhere.

A distribution of classes (pixels), each with a specific color (where 0 is black):

Distribution of classes (pixels) in Y_tensor:
Class 0: 852,923 pixels (88.08%)
Class 1: 18,862 pixels (1.95%)
Class 2: 12,489 pixels (1.29%)
Class 3: 13,746 pixels (1.42%)
Class 4: 16,426 pixels (1.70%)
Class 5: 8,142 pixels (0.84%)
Class 6: 6,035 pixels (0.62%)
Class 7: 14,794 pixels (1.53%)
Class 8: 21,614 pixels (2.23%)
Class 9: 3,369 pixels (0.35%)

<Figure size 600x300 with 2 Axes><img width="551" height="308" alt="image" src="https://github.com/user-attachments/assets/dd34f838-5e48-4ea2-8d28-d9c8bc2a7beb" />


It was therefore necessary to penalize the black color and reconstruct the grids from the tensors, then re-train the models.

Black was no longer omnipresent, but gray began to take its place.

<Figure size 600x300 with 2 Axes><img width="555" height="308" alt="image" src="https://github.com/user-attachments/assets/fba127f3-888c-43b3-85d2-99f6ff0ac864" />

<Figure size 600x300 with 2 Axes><img width="555" height="308" alt="image" src="https://github.com/user-attachments/assets/5199c380-1679-40cc-ad3e-6c366b66c6cb" />

Here, tasks 00d62c1b and 00dbd492 (Reminder: grid resizing was not done, which is normal as it wasn't worked on, but we can see that the measurements are correct).

We prepare the training again, restructuring the whole set, still penalizing black, re-creating lookup tables where we filter valid indices, and then deepen the model by using all the training data, followed by extracting symbolic features for all the inputs of all the training tasks.

We then merge the architectures and retrain everything for 40 Epochs.

Subsequent steps will involve continuously adding transformation verifications, conversions to tensors, and optimized training.

We end up with more robust handling of variable layouts.

<Figure size 1500x600 with 2 Axes><img width="1478" height="617" alt="image" src="https://github.com/user-attachments/assets/10c0b2cc-dee9-49c5-a87f-224bb1f5e05c" />


Some predictions are not yet perfect but come very close to the expected result!

For example with 1d0a4b61, db695cfb, and 4a21e3da (which, apart from the resizing, is predicted perfectly).

<Figure size 1500x1500 with 3 Axes><img width="817" height="1535" alt="image" src="https://github.com/user-attachments/assets/1f61ef5b-25df-4491-aaf7-ca971a857349" />


<Figure size 1500x1500 with 3 Axes><img width="813" height="1535" alt="image" src="https://github.com/user-attachments/assets/2dcc0826-2819-44a6-9942-586696b9c85d" />


<Figure size 1500x1500 with 6 Axes><img width="957" height="1535" alt="image" src="https://github.com/user-attachments/assets/c68cde73-5936-4f04-8954-835c94ffb5b3" />


Other tasks like 42a50994 are poorly predicted. This demonstrates that the model is not yet versatile.

<Figure size 1500x1500 with 3 Axes><img width="817" height="1535" alt="image" src="https://github.com/user-attachments/assets/7af16258-db34-4857-b5a5-15e6e0019403" />

