import("csvfast", "Table")

-- CONFIG
local FEATURES = {
	log_price = true,

	accommodates = true,
	bathrooms = true,
	bedrooms = true,
	beds = true,

	number_of_reviews = true,
	review_scores_rating = true,

	latitude = true,
	longitude = true,

	cleaning_fee = true,
	instant_bookable = true,
	host_identity_verified = true,
	room_type = true,
	property_type = true
}

local COLUMN_ORDER = {
	"accommodates", "bathrooms", "bedrooms", "beds", "reviews", 
	"rating", "lat", "lon", "cleaning_fee", "instant_bookable", 
	"host_verified", "room_entire", "room_private", "room_shared", 
	"is_apartment", "log_price"
}

-- UTILS
local function bool(v)
	if v == nil then return 0 end
	local s = string.lower(tostring(v))
	if s == "1" or s == "t" or s == "true" or s == "yes" then
		return 1
	end
	return 0
end

local function clip(v, a, b)
	if v ~= v then return v end -- NaN protection
	if v < a then return a end
	if v > b then return b end
	return v
end

local function safe_num(v)
	if v ~= v then return nil end
	if v == nil then return nil end
	if type(v) == "number" then return v end
	return tonumber(v)
end

local function clean_str(s)
    if not s then return "" end
    return tostring(s):lower():gsub('"', ''):gsub('^%s*(.-)%s*$', '%1')
end

local preworker = {}

function preworker.start(workerId, totalWorkers, base, prefix)
	-- MEAN IMPUTATION GLOBAL
	local fields = {"bathrooms", "bedrooms", "beds", "review_scores_rating"}
	local sum, count = {}, {}

	for _, f in ipairs(fields) do
		sum[f], count[f] = 0, 0
	end

	csvfast.each(base, function(r)
		for _, f in ipairs(fields) do
			local v = safe_num(r[f])
			if v then
				sum[f] = sum[f] + v
				count[f] = count[f] + 1
			end
		end
	end, {
		schema = FEATURES
	})

	local mean = {}
	for _, f in ipairs(fields) do
		mean[f] = (count[f] > 0) and (sum[f] / count[f]) or 0
	end

	-- SHARDING
	local total = csvfast.count_rows(base)
	local chunk = math.floor(total / totalWorkers)

	local start = (workerId - 1) * chunk
	local limit = chunk

	if workerId == totalWorkers then
		limit = total - start
	end

	print(string.format("[Worker %d] start=%d limit=%d", workerId, start, limit))

	-- DATASET
	local train = Table.new()
	local idx = 0

	csvfast.each(base, function(r)
		if idx < start or idx >= start + limit then
			idx = idx + 1
			return
		end
		idx = idx + 1

		if idx <= 5 then
			Table.stream(r, true, string.format("debug_row_%d.lua", idx))
		end

		-- target
		local y = safe_num(r.log_price) or 0
		local acc = safe_num(r.accommodates) or 0
		local bath = safe_num(r.bathrooms) or 0
		local bed = safe_num(r.bedrooms) or 0
		local beds = safe_num(r.beds) or 0
		local rat = safe_num(r.review_scores_rating) or 0

		-- Acceso más seguro usando r["nombre"] por si r.nombre falla
		local room = clean_str(r["room_type"])
		local prop = clean_str(r["property_type"])

		local row = {
			log_price    = y,
			accommodates = acc,
			bathrooms    = bath,
			bedrooms     = bed,
			beds         = beds,
			reviews      = safe_num(r["number_of_reviews"]) or 0,
			rating       = rat,
			lat          = safe_num(r["latitude"]) or 0,
			lon          = safe_num(r["longitude"]) or 0,

			-- Revisa que r.cleaning_fee no sea nil antes de pasar a bool()
			cleaning_fee     = bool(r["cleaning_fee"]),
			instant_bookable = bool(r["instant_bookable"]),
			host_verified    = bool(r["host_identity_verified"]),

			-- Comparación de strings
			room_entire  = (room:find("entire")) and 1 or 0,
			room_private = (room:find("private")) and 1 or 0,
			room_shared  = (room:find("shared")) and 1 or 0,
			is_apartment = (prop:find("apartment")) and 1 or 0
		}
		train:iput(row)
	end, {
		schema = FEATURES,
		start = start,
		limit = limit
	})

	print("[Worker "..workerId.."] train:", train:len())

	-- EXPORT
	local function export(dataset, outprefix)
		local output_table = {}
		for _, colName in ipairs(COLUMN_ORDER) do
			output_table[colName] = {}
		end

		-- dataset es tu objeto 'train' (csvfast.Table)
		for i = 1, dataset:len() do
			local r = dataset[i] 
			for _, colName in ipairs(COLUMN_ORDER) do
				-- Asegúrate de que r[colName] no sea nil
				local val = r[colName] or 0
				table.insert(output_table[colName], val)
			end
		end

		-- IMPORTANTE: Verifica si el archivo se escribe bien
		csvfast.save_columns(output_table, outprefix .. "_" .. workerId .. ".csv", COLUMN_ORDER)
	end

	export(train, "data/dataset_train")
	print("[Worker "..workerId.."] EXPORT OK")
end

return preworker
