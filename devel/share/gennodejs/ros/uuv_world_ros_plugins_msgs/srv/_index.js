
"use strict";

let SetCurrentDirection = require('./SetCurrentDirection.js')
let GetOriginSphericalCoord = require('./GetOriginSphericalCoord.js')
let SetCurrentModel = require('./SetCurrentModel.js')
let GetCurrentModel = require('./GetCurrentModel.js')
let SetOriginSphericalCoord = require('./SetOriginSphericalCoord.js')
let SetCurrentVelocity = require('./SetCurrentVelocity.js')
let TransformToSphericalCoord = require('./TransformToSphericalCoord.js')
let TransformFromSphericalCoord = require('./TransformFromSphericalCoord.js')

module.exports = {
  SetCurrentDirection: SetCurrentDirection,
  GetOriginSphericalCoord: GetOriginSphericalCoord,
  SetCurrentModel: SetCurrentModel,
  GetCurrentModel: GetCurrentModel,
  SetOriginSphericalCoord: SetOriginSphericalCoord,
  SetCurrentVelocity: SetCurrentVelocity,
  TransformToSphericalCoord: TransformToSphericalCoord,
  TransformFromSphericalCoord: TransformFromSphericalCoord,
};
