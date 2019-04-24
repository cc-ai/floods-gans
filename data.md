# Data

Here is a list of datasets we've used and there structure. Pictures are "open-source", under various licences. Reach out if you need specifics

|      Dataset name      |                                     link                                     | description                                                                                                                                     |
| :--------------------: | :--------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| 27.02.FloodingData.zip | [GDrive](https://drive.google.com/file/d/1QFjBtzZ6XpLEXgqjd9_Bdz__QmV5m-TW)  | 486 pictures of "unaltered" houses,   539 pictures involving floods (+ Venice)  of   which we consider 159 to be of good quality to train a GAN |
|  10.03CroppedData.zip  | [GDrive](https://drive.google.com/open?id=1yORz3AmiFI8GKwFdEZ3qFqbaEVN37O41) | Same pictures as `27.02.FloodingData` but houses sorted according to presence of Grass or Street                                                |
| 11.03AllHousesAugmented.zip | [GDrive](https://drive.google.com/file/d/1K195Qz_dsqqMIlbqNkiZ8zoUik5W7F5G)  | 3,562 pictures of houses, augmented based on all of the images of unaltered houses from 27.02.FloodingData by reducing dimensionality, cropping and flipping horizontally  |
| 11.03AllHousesFlooded.zip  | [GDrive](https://drive.google.com/file/d/1aY27R6tdoJzk6FLVPsacb0p6k1gMDThL) |  3,539 pictures of flooding, augmented based on all of the flooding images from 27.02.FloodingData by reducing dimensionality, cropping and flipping horizontally   |
| 11.03BestFloodAugmented    | [GDrive](https://drive.google.com/file/d/1Z8ns2Y2xK_ZUXimG5zRVFmlyJGu2bUoy) | 1,175 pictures of flooding, augmented based on the best flooding images from 27.02.FloodingData by reducing dimensionality, cropping and flipping horizontally  |

Description of every image in `27.02.FloodingData` can be found in [27.02.FloodingData_image_descriptions](27.02.FloodingData_image_descriptions.txt)