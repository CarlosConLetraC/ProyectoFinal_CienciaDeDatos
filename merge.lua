import("csvfast", "json", "cml", "cstats", "task")

-- =========================
-- SHARD DISCOVERY
-- =========================
local function list_files(prefix)
	local p = io.popen("ls data/" .. prefix .. "_*.csv 2>/dev/null")
	if not p then return {} end

	local files = {}

	for file in p:lines() do
		local n = file:match(prefix .. "_(%d+)%.csv")
		if n then
			files[#files+1] = {
				path = file,
				idx = tonumber(n)
			}
		end
	end

	p:close()

	table.sort(files, function(a, b)
		return a.idx < b.idx
	end)

	return files
end

local function bool(v)
    if v == nil then return 0 end
    local s = string.lower(tostring(v))
    if s == "1" or s == "t" or s == "true" or s == "yes" then
        return 1
    end
    return 0
end

local train_shards = list_files("dataset_train")
local test_shards  = list_files("dataset_test")

print("Train shards:", #train_shards)
print("Test shards :", #test_shards)

if #train_shards == 0 then
	error("no train shards found")
end

-- =========================
-- STORAGE
-- =========================
local train = {}
local test  = {}

-- =========================
-- SCHEMA
-- =========================
local schema = {
	log_price = true,
	accommodates = true,
	bathrooms = true,
	bedrooms = true,
	beds = true,
	reviews = true,
	rating = true,
	lat = true,
	lon = true,
	cleaning_fee = true,
	instant_bookable = true,
	host_verified = true,
	room_entire = true,
	room_private = true,
	room_shared = true,
	is_apartment = true
}

-- =========================
-- LOADER (SAFE)
-- =========================
local function loader(shards, target, label)
	for _, s in ipairs(shards) do
		print("["..label.."] loading:", s.path)

		csvfast.each(s.path, function(r)
			if not r then return end

			local y = r.log_price
			if not y or y ~= y then return end

			local lat = r.lat or r.latitude
			local lon = r.lon or r.longitude
			if not lat or not lon then return end

			target[#target+1] = {
				log_price = y,
				accommodates = r.accommodates,
				bathrooms = r.bathrooms,
				bedrooms = r.bedrooms,
				beds = r.beds,
				reviews = r.reviews,
				rating = r.rating,
				lat = lat,
				lon = lon,
				cleaning_fee = r.cleaning_fee,
				instant_bookable = bool(r.instant_bookable),
				host_verified = bool(r.host_verified),
				room_entire = bool(r.room_entire),
				room_private = bool(r.room_private),
				room_shared = bool(r.room_shared),
				is_apartment = bool(r.is_apartment)
			}
		end, { schema = schema })

		task.wait(0)
	end

	print("["..label.."] DONE")
end

task.spawn(loader, train_shards, train, "TRAIN")
task.spawn(loader, test_shards, test, "TEST")

while task.step() do end

print("Train size:", #train)
print("Test size :", #test)

-- =========================
-- IMPORTANT FIX: SINGLE SPLIT ONLY
-- =========================
local function split(data, ratio)
	local n = #data
	if n < 10 then
		error("dataset too small for split: " .. n)
	end

	local split = math.floor(n * ratio)

	if split < 2 then split = 2 end
	if split >= n then split = n - 2 end

	local trainSet = {}
	local testSet  = {}

	for i = 1, split do trainSet[#trainSet+1] = data[i] end
	for i = split+1, n do testSet[#testSet+1] = data[i] end

	return trainSet, testSet
end

-- fallback if needed
if #test == 0 then
	print("[WARN] no test shards -> internal split 80/20")
	train, test = split(train, 0.8)
end

-- =========================
-- FEATURES
-- =========================
local features = {"accommodates", "bathrooms", "bedrooms", "beds", "reviews", "rating", "lat", "lon", "cleaning_fee", "instant_bookable", "host_verified", "room_entire", "room_private", "room_shared", "is_apartment"}

-- =========================
-- MODEL
-- =========================
local model = cml.LinearRegression({
    features = features,
    target = "log_price"
})
print("[MODEL] Loading data into C++ matrix...")
model:load(train)

print("[MODEL] Computing statistics and normalizing...")
model:normalize()

model:fit(train, 0.05, 5000, 0.8, 0.02)

local mse = model:mse(test)
local r2  = model:r2(test)

-- Para MAE, si no lo implementamos en C++, seguimos usando cstats
local y_true, y_pred = {}, {}
for i = 1, #test do
	y_true[i] = test[i].log_price
	y_pred[i] = model:predict(test[i])
end
local mae = cstats.mae(y_true, y_pred)

print("===== METRICS (C++ API) =====")
print("MAE:", mae)
print("MSE:", mse)
print("R2 :", r2)

-- EXPORTACION (Pesos des-normalizados)
local out = {
	meta = {
		train = #train,
		test = #test,
		features = features
	},
	metrics = {
		mae = mae,
		mse = mse,
		r2 = r2
	},
	model = model:export()
}

local f = assert(io.open("data/model_final.json", "w"))
f:write(json.encode(out))
f:close()

print("MODEL OK -> data/model_final.json")