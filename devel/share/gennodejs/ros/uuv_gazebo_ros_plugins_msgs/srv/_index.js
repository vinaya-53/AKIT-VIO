
"use strict";

let GetFloat = require('./GetFloat.js')
let SetUseGlobalCurrentVel = require('./SetUseGlobalCurrentVel.js')
let GetModelProperties = require('./GetModelProperties.js')
let SetFloat = require('./SetFloat.js')
let GetThrusterConversionFcn = require('./GetThrusterConversionFcn.js')
let SetThrusterState = require('./SetThrusterState.js')
let GetThrusterState = require('./GetThrusterState.js')
let GetThrusterEfficiency = require('./GetThrusterEfficiency.js')
let SetThrusterEfficiency = require('./SetThrusterEfficiency.js')
let GetListParam = require('./GetListParam.js')

module.exports = {
  GetFloat: GetFloat,
  SetUseGlobalCurrentVel: SetUseGlobalCurrentVel,
  GetModelProperties: GetModelProperties,
  SetFloat: SetFloat,
  GetThrusterConversionFcn: GetThrusterConversionFcn,
  SetThrusterState: SetThrusterState,
  GetThrusterState: GetThrusterState,
  GetThrusterEfficiency: GetThrusterEfficiency,
  SetThrusterEfficiency: SetThrusterEfficiency,
  GetListParam: GetListParam,
};
