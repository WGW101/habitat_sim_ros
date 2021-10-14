# habitat\_sim\_ros

## Subscribed Topics
- `~cmd_vel`                        : `geometry_msgs/Twist`

## Published Topics
- `~camera/rgb/image_raw`           : `sensor_msgs/Image`
- `~camera/rgb/camera_info`         : `sensor_msgs/CameraInfo` : latched
- `~camera/depth/image_raw`         : `sensor_msgs/Image`
- `~camera/depth/camera_info`       : `sensor_msgs/CameraInfo` : latched
- `~scan`                           : `sensor_msgs/LaserScan`
- `~map`                            : `nav_msgs/OccupancyGrid` : *optional*, latched
- `/clock`                          : `rosgraph_msgs/Clock` : *optional*

## Broadcasted Transforms
- `map` → `odom` : *optional*
- `odom` → `base_footprint`
- `base_footprint` → `scan_frame` : static
- `base_footprint` → `camera/rgb/optical_frame` : static
- `base_footprint` → `camera/depth/optical_frame` : static

## Provided Services
- `~load_scene`                     : `habitat_sim_ros/LoadScene`
    * `scene_id`                    : `string`
- `~respawn_agent`                  : `habitat_sim_ros/RespawnAgent`
    * `pose`                        : `geometry_msgs/Pose`
- `~reset`                          : `habitat_sim_ros/Reset`

## Parameters
- `/use_sim_time`                   : `bool`  : default = false
- `~sim/rate`                       : `float` : default = 60.0Hz
- `~sim/allow_sliding`              : `bool`  : default = true
- `~sim/scene_id`                   : `str`   : **required!**
- `~sim/seed`                       : `int`   : *optional*

- `~agent/height`                   : `float` : default = 0.8m
- `~agent/radius`                   : `float` : default = 0.2m
- `~agent/start_position/random`    : `bool`  : default = true
- `~agent/start_position/x`         : `float` : default = 0.0m
- `~agent/start_position/y`         : `float` : default = 0.0m
- `~agent/start_position/z`         : `float` : default = 0.0m
- `~agent/start_orientation/random` : `bool`  : default = true
- `~agent/start_orientation/yaw`    : `float` : default = 0.0°

- `~agent/ctrl/timeout`             : `float` : default = 1.0s
- `~agent/ctrl/linear/max_vel`      : `float` : default = 1.0m/s
- `~agent/ctrl/linear/reverse`      : `bool`  : default = true
- `~agent/ctrl/linear/max_acc`      : `float` : default = 0.2m/s²
- `~agent/ctrl/linear/max_brk`      : `float` : default = 0.4m/s²
- `~agent/ctrl/linear/noise/bias`   : `float` : default = 0.0m/s
- `~agent/ctrl/linear/noise/stdev`  : `float` : default = 0.0m/s
- `~agent/ctrl/angular/max_vel`     : `float` : default = 1.5rad/s
- `~agent/ctrl/angular/max_acc`     : `float` : default = 0.5rad/s²
- `~agent/ctrl/angular/max_brk`     : `float` : default = 0.8rad/s²
- `~agent/ctrl/angular/noise/bias`  : `float` : default = 0.0rad/s
- `~agent/ctrl/angular/noise/stdev` : `float` : default = 0.0rad/s

- `~map/enabled`                    : `bool`  : default = true
- `~map/resolution`                 : `float` : default = 0.02m/px

- `~odom/linear/noise/bias`         : `float` : default = 0.0 TODO: unit
- `~odom/linear/noise/stdev`        : `float` : default = 0.0 TODO: unit
- `~odom/angular/noise/bias`        : `float` : default = 0.0 TODO: unit
- `~odom/angular/noise/stdev`       : `float` : default = 0.0 TODO: unit

- `~scan/position/x`                : `float` : default = 0.0m
- `~scan/position/y`                : `float` : default = 0.0m
- `~scan/position/z`                : `float` : default = 0.45m
- `~scan/range/min`                 : `float` : default = 0.0m
- `~scan/range/max`                 : `float` : default = 10.0m
- `~scan/num_rays`                  : `int`   : default = 360
- `~scan/noise/bias`                : `float` : default = 0.0m
- `~scan/noise/stdev`               : `float` : default = 0.0m

- `~rgb/position/x`                 : `float` : default = 0.0m
- `~rgb/position/y`                 : `float` : default = 0.0m
- `~rgb/position/z`                 : `float` : default = 0.6m
- `~rgb/orientation/tilt`           : `float` : default = 0.0°
- `~rgb/width`                      : `int`   : default = 320
- `~rgb/height`                     : `int`   : default = 240
- `~rgb/hfov`                       : `int`   : default = 60°

- `~depth/position/x`               : `float` : default = 0.0m
- `~depth/position/y`               : `float` : default = 0.0m
- `~depth/position/z`               : `float` : default = 0.6m
- `~depth/orientation/tilt`         : `float` : default = 0.0°
- `~depth/width`                    : `int`   : default = 320
- `~depth/height`                   : `int`   : default = 240
- `~depth/hfov`                     : `int`   : default = 60°

# TODO
- Normalize odom noise (currently drift at every update, regardless of frametime...)
- Improve doc (this README)
- Support for `dynamic_reconfigure`
