
"use strict";

let StartTrajectory = require('./StartTrajectory.js')
let SetSMControllerParams = require('./SetSMControllerParams.js')
let InitWaypointsFromFile = require('./InitWaypointsFromFile.js')
let ResetController = require('./ResetController.js')
let GetPIDParams = require('./GetPIDParams.js')
let InitRectTrajectory = require('./InitRectTrajectory.js')
let InitHelicalTrajectory = require('./InitHelicalTrajectory.js')
let GoToIncremental = require('./GoToIncremental.js')
let AddWaypoint = require('./AddWaypoint.js')
let IsRunningTrajectory = require('./IsRunningTrajectory.js')
let SwitchToManual = require('./SwitchToManual.js')
let InitCircularTrajectory = require('./InitCircularTrajectory.js')
let SetMBSMControllerParams = require('./SetMBSMControllerParams.js')
let GetWaypoints = require('./GetWaypoints.js')
let GetSMControllerParams = require('./GetSMControllerParams.js')
let SetPIDParams = require('./SetPIDParams.js')
let SwitchToAutomatic = require('./SwitchToAutomatic.js')
let GoTo = require('./GoTo.js')
let InitWaypointSet = require('./InitWaypointSet.js')
let GetMBSMControllerParams = require('./GetMBSMControllerParams.js')
let Hold = require('./Hold.js')
let ClearWaypoints = require('./ClearWaypoints.js')

module.exports = {
  StartTrajectory: StartTrajectory,
  SetSMControllerParams: SetSMControllerParams,
  InitWaypointsFromFile: InitWaypointsFromFile,
  ResetController: ResetController,
  GetPIDParams: GetPIDParams,
  InitRectTrajectory: InitRectTrajectory,
  InitHelicalTrajectory: InitHelicalTrajectory,
  GoToIncremental: GoToIncremental,
  AddWaypoint: AddWaypoint,
  IsRunningTrajectory: IsRunningTrajectory,
  SwitchToManual: SwitchToManual,
  InitCircularTrajectory: InitCircularTrajectory,
  SetMBSMControllerParams: SetMBSMControllerParams,
  GetWaypoints: GetWaypoints,
  GetSMControllerParams: GetSMControllerParams,
  SetPIDParams: SetPIDParams,
  SwitchToAutomatic: SwitchToAutomatic,
  GoTo: GoTo,
  InitWaypointSet: InitWaypointSet,
  GetMBSMControllerParams: GetMBSMControllerParams,
  Hold: Hold,
  ClearWaypoints: ClearWaypoints,
};
